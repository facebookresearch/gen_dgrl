# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import itertools
import json
import os
import pathlib
import sys
from typing import Dict, List

import submitit

from train_scripts import arguments
from utils.utils import merge_two_dicts, permutate_params_and_merge


def generate_all_params(grid, defaults, num_trials=1, start_index=0) -> List[Dict[str, any]]:
    grid = merge_two_dicts(
        grid,
        {
            "xpid": [n for n in range(start_index, start_index + num_trials)],
        },
    )
    permutations = permutate_params_and_merge(grid, defaults)
    return [
        merge_two_dicts(
            perm,
            {"seed": perm["seed"] + perm["xpid"]},
        )
        for perm in permutations
    ]


def generate_command(params: Dict[str, any], newlines: bool, xpid_generator, algo: str) -> str:
    if xpid_generator:
        params["xpid"] = xpid_generator(params, algo) + f"_{params['xpid']}"

    separator = " \\\n" if newlines else " "
    header = f"python -m {args.module_name}"
    cmd = [header] + [f"--{k}={vi}" for k, v in params.items() for vi in (v if isinstance(v, list) else [v])]
    return separator.join(cmd)


def generate_slurm_commands(params: Dict[str, any], module_name: str, xpid_generator, algo: str) -> List[str]:
    if xpid_generator:
        params["xpid"] = xpid_generator(params, algo) + f"_{params['xpid']}"

    header = [sys.executable, "-m", module_name]
    args = itertools.chain(
        *[(f"--{k}", str(vi)) for k, v in params.items() for vi in (v if isinstance(v, list) else [v])]
    )
    return header + list(args)


def xpid_from_params(p, algo: str = "") -> str:
    env_prefix = f"{p['env_name']}-{p['distribution_mode']}-{p['num_levels']}"

    """python function which converts long integers into short strings
        Example: 1000000 -> 1M, 1000 -> 1K, etc.
    """

    def short_int(n):
        if n >= 1_000_000:
            return f"{int(n/1_000_000)}M"
        elif n >= 1000:
            return f"{int(n/1000)}K"
        else:
            return f"{n}"

    if "dataset_size" in p:
        env_prefix = f"{env_prefix}-d{short_int(p['dataset_size'])}"

    if algo in ["bc", "cql", "bcq", "iql", "dt", "bct"]:
        algo_prefix = f"{algo}-p{p['percentile']}-lr{p['lr']}-bs{p['batch_size']}-{p['agent_model']}"
        if algo in ["cql", "bcq", "iql"]:
            algo_prefix = f"{algo_prefix}-tau{p['tau']}-tuf{p['target_update_freq']}"
        if algo == "cql":
            algo_prefix = f"{algo_prefix}-a{p['cql_alpha']}"
        elif algo == "bcq":
            assert p["agent_model"] in ["bcq", "bcqresnetbase"]
            algo_prefix = f"{algo_prefix}-t{p['bcq_threshold']}"
            if p["agent_model"] == "bcqresnetbase":
                algo_prefix = f"{algo_prefix}-res"
        elif algo == "iql":
            algo_prefix = f"{algo_prefix}-t{p['iql_temperature']}-e{p['iql_expectile']}"
        elif algo in ["dt", "bct"]:
            algo_prefix = f"{algo_prefix}-cl{p['dt_context_length']}-er{p['dt_eval_ret']}"
    elif algo == "ppo":
        algo_prefix = (
            f"{algo}-lr{p['lr']}-epoch{p['ppo_epoch']}-mb{p['num_mini_batch']}"
            + f"-v{p['value_loss_coef']}-ha{p['entropy_coef']}"
        )
    else:
        algo_prefix = f"{algo}-lr{p['lr']}"
    
    if "early_stop" in p and p['early_stop']:
        algo_prefix = f"{algo_prefix}-es"

    return f"{env_prefix}-{algo_prefix}"


class LaunchExperiments:
    def __init__(self):
        pass

    def launch_experiment_and_remoteenv(self, experiment_args):
        # imports and definition are inside of function because of submitit
        import multiprocessing as mp

        def launch_experiment(experiment_args):
            import subprocess

            subprocess.call(
                generate_slurm_commands(
                    params=experiment_args,
                    module_name=args.module_name,
                    xpid_generator=xpid_from_params,
                    algo=args.grid_config,
                )
            )

        experiment_process = mp.Process(target=launch_experiment, args=[experiment_args])
        self.process = experiment_process
        experiment_process.start()
        experiment_process.join()

    def __call__(self, experiment_args):
        self.launch_experiment_and_remoteenv(experiment_args)

    def checkpoint(self, experiment_args) -> submitit.helpers.DelayedSubmission:
        self.process.terminate()
        return submitit.helpers.DelayedSubmission(LaunchExperiments(), experiment_args)


def schedule_slurm_jobs(all_params: List[Dict[str, any]], job_name: str, dry_run: bool) -> None:
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")

    rootdir = os.path.expanduser(f"~/slurm/{job_name}")
    submitit_dir = os.path.expanduser(f"~/slurm/{job_name}/{now}")
    executor = submitit.SlurmExecutor(folder=submitit_dir)
    os.makedirs(submitit_dir, exist_ok=True)

    symlink = os.path.join(rootdir, "latest")
    if os.path.islink(symlink):
        os.remove(symlink)
    if not os.path.exists(symlink):
        os.symlink(submitit_dir, symlink)
        print("Symlinked experiment directory: %s", symlink)

    executor.update_parameters(
        # examples setup
        partition="learnlab",
        # partition="prioritylab",
        comment="Neurips 2023 submission",
        time=1 * 72 * 60,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name=job_name,
        mem="160GB",
        cpus_per_task=10,
        gpus_per_node=1,
        constraint="volta32gb",
        array_parallelism=1024,
    )

    if not dry_run:
        jobs = executor.map_array(LaunchExperiments(), all_params)

        for job in jobs:
            print("Submitted with job id: ", job.job_id)
            print(f"stdout -> {submitit_dir}/{job.job_id}_0_log.out")
            print(f"stderr -> {submitit_dir}/{job.job_id}_0_log.err")

        print(f"Submitted {len(jobs)} jobs! \n")

        print(submitit_dir)


if __name__ == "__main__":
    args = arguments.parser.parse_args()

    # Default Params
    defaults = json.load(
        open(os.path.expandvars(os.path.expanduser(os.path.join("configs", args.base_config, "default.json"))))
    )
    if args.checkpoint:
        defaults["resume"] = True

    if args.wandb_project:
        defaults["wandb_project"] = args.wandb_project

    if args.wandb_base_url:
        defaults["wandb_base_url"] = args.wandb_base_url
    if args.wandb_api_key:
        defaults["wandb_api_key"] = args.wandb_api_key
    if args.wandb_entity:
        defaults["wandb_entity"] = args.wandb_entity

    # Generate all parameter combinations within grid, using defaults for fixed params
    config = json.load(
        open(
            os.path.expandvars(
                os.path.expanduser(os.path.join("configs", args.base_config, "grids", args.grid_config + ".json"))
            )
        )
    )
    all_params = generate_all_params(config["grid"], defaults, args.num_trials, args.start_index)
    print(f"About to generate {len(all_params)} commands! \n")

    # Action
    if args.action == "PRINT":
        cmds = [generate_command(params, args.new_line, xpid_from_params, args.grid_config) for params in all_params]

        print("Generated Commands: \n")
        [print(f"{cmd} \n") for cmd in cmds]

    elif args.action == "SAVE":
        cmds = [generate_command(params, args.new_line, xpid_from_params, args.grid_config) for params in all_params]

        filename = f"{args.grid_config}_commands.txt"
        pathlib.Path(filename).touch()
        with open(filename, "w") as f:
            [f.write(f"{cmd} \n\n") for cmd in cmds]

        print(f"Generated commands are stored in {filename}.")

    elif args.action == "SLURM":
        schedule_slurm_jobs(all_params, args.job_name, args.dry_run)

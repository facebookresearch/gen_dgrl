import argparse
import json
import os
from typing import List

from utils.utils import permutate_params_and_merge


def generate_train_cmds(
    params,
    num_trials=1,
    start_index=0,
    newlines=False,
    xpid_generator=None,
    algo="",
    xpid_prefix="",
    is_single=False,
) -> List[str]:
    separator = " \\\n" if newlines else " "

    cmds = []

    if xpid_generator:
        params["xpid"] = xpid_generator(params, xpid_prefix, algo, is_single)

    for t in range(num_trials):
        trial_idx = t + start_index
        params["seed"] += trial_idx

        cmd = [f"python -m {args.module_name}"] + [
            f"--{k}={vi}_{trial_idx}" if k == "xpid" else f"--{k}={vi}"
            for k, v in params.items()
            for vi in (v if isinstance(v, list) else [v])
        ]

        cmds.append(separator.join(cmd))

    return cmds


def parse_args():
    parser = argparse.ArgumentParser(description="Make commands")

    # Principle arguments
    parser.add_argument(
        "--base_config",
        type=str,
        choices=["offline", "online"],
        default="offline",
        help="Base config where parameters are set with default values, and may be replaced by sweeping.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="grid",
        help="Directory where the configs are present",
    )
    parser.add_argument(
        "--grid_config",
        type=str,
        help="Name of the .json config for hyperparameter search-grid.",
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="Number of seeds to be used for each hyperparameter setting."
    )
    parser.add_argument(
        "--module_name",
        type=str,
        choices=["offline.train_offline_agent", "online.trainer", "offline.single_level_train_offline_agent"],
        default="offline.train_offline_agent",
        help="Name of module to be used in the generated commands. "
        + "The result will be like 'python -m <MODULE_NAME> ...'",
    )
    parser.add_argument("--start_index", default=0, type=int, help="Starting trial index of xpid runs")
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="If true, a boolean flag 'resume' will be put in the generated commands, "
        + "which indicates offline training to resume from a given checkpoint.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="single level training"
    )

    # Misc
    parser.add_argument(
        "--new_line",
        action="store_true",
        help="If true, print generated commands with arguments separated by new line; "
        + "otherwise arguments will be separated by space.",
    )
    parser.add_argument("--count", action="store_true", help="If true, print number of generated commands at the end.")

    # wandb
    parser.add_argument("--wandb_base_url", type=str, default=None, help="wandb base url")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="wandb api key")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")

    return parser.parse_args()


def xpid_from_params(p, prefix="", algo="", is_single=False):
    env_prefix = f"{p['env_name']}-{p['distribution_mode']}-{p['num_levels']}"
    
    """python function which converts long integers into short strings
        Example: 1000000 -> 1M, 1000 -> 1K, etc.
    """
    def short_int(n):
        if n >= 1000000:
            return f"{int(n/1000000)}M"
        elif n >= 1000:
            return f"{int(n/1000)}K"
        else:
            return f"{n}"
        
    if "dataset_size" in p:
        env_prefix = f"{env_prefix}-d{short_int(p['dataset_size'])}"

    if algo in ["bc", "cql", "bcq", "iql", "dt", "bct"]:
        algo_prefix = f"{algo}-p{p['percentile']}-lr{p['lr']}-bs{p['batch_size']}-{p['agent_model']}"
        if algo in ["cql", "bcq", "iql"]:
            algo_prefix = f"{algo_prefix}-tuf{p['target_update_freq']}"
            if p['perform_polyak_update']:
                algo_prefix = f"{algo_prefix}-polyak-tau{p['tau']}"
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
        
    if is_single:
        algo_prefix = f"{algo_prefix}-single"
        if p['capacity_type']=="transitions":
            algo_prefix = f"{algo_prefix}-t"
        elif p['capacity_type']=="episodes":
            algo_prefix = f"{algo_prefix}-e"
        algo_prefix = f"{algo_prefix}-{p['threshold_metric']}"

    return f"{env_prefix}-{algo_prefix}"


if __name__ == "__main__":
    args = parse_args()

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
                os.path.expanduser(os.path.join("configs", args.base_config, args.dir, args.grid_config + ".json"))
            )
        )
    )
    all_params = permutate_params_and_merge(config["grid"], defaults=defaults)

    # Print all commands
    xpid_prefix = "" if "xpid_prefix" not in config else config["xpid_prefix"]
    for p in all_params:
        cmds = generate_train_cmds(
            p,
            num_trials=args.num_trials,
            start_index=args.start_index,
            newlines=args.new_line,
            xpid_generator=xpid_from_params,
            algo=args.grid_config,
            xpid_prefix=xpid_prefix,
            is_single=args.single,
        )

        for c in cmds:
            print(c + "\n")

    if args.count:
        print(f"Generated {len(all_params) * args.num_trials} commands.")

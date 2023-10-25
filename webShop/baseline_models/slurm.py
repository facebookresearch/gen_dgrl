# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import sys

import submitit
from absl import app, flags
from coolname import generate_slug

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "~/cmds.txt", "Path to list of commands to run.")
flags.DEFINE_string("name", "anyslurm", "Experiment name.")
flags.DEFINE_boolean("debug", False, "Only debugging output.")
flags.DEFINE_string("partition", "learnlab", "partition name.")
flags.DEFINE_string("algo", "il", "algo_name.")

def arg2str(k, v):
    if isinstance(v, bool):
        if v:
            return ("--%s" % k,)
        else:
            return ""
    else:
        return ("--%s" % k, str(v))


class LaunchExperiments:
    def __init__(self, algo):
        self.algo = algo

    def launch_experiment_and_remotenv(self, experiment_args):
        # imports and definition are inside of function because of submitit
        import multiprocessing as mp

        def launch_experiment(experiment_args):
            import itertools
            import subprocess

            python_exec = sys.executable
            args = itertools.chain(*[arg2str(k, v) for k, v in experiment_args.items()])

            if self.algo == "test":
                subprocess.call([python_exec, "-m", "test"] + list(args))
            else:
                subprocess.call([python_exec, "-m", f"train_choice_{self.algo}"] + list(args))

        experiment_process = mp.Process(target=launch_experiment, args=[experiment_args])
        self.process = experiment_process
        experiment_process.start()
        experiment_process.join()

    def __call__(self, experiment_args):
        self.launch_experiment_and_remotenv(experiment_args)

    def checkpoint(self, experiment_args) -> submitit.helpers.DelayedSubmission:
        self.process.terminate()
        experiment_args["checkpoint"] = True
        return submitit.helpers.DelayedSubmission(LaunchExperiments(self.is_offline), experiment_args)


def main(argv):
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")

    rootdir = os.path.expanduser(f"~/slurm/{FLAGS.name}")
    submitit_dir = os.path.expanduser(f"~/slurm/{FLAGS.name}/{now}")
    executor = submitit.SlurmExecutor(folder=submitit_dir)
    os.makedirs(submitit_dir, exist_ok=True)

    symlink = os.path.join(rootdir, "latest")
    if os.path.islink(symlink):
        os.remove(symlink)
    if not os.path.exists(symlink):
        os.symlink(submitit_dir, symlink)
        print("Symlinked experiment directory: %s", symlink)

    all_args = list()

    with open(os.path.expanduser(FLAGS.path), "r") as f:
        cmds = "".join(f.readlines()).split("\n\n")
        cmds = [cmd.split("\\\n")[1:] for cmd in cmds]
        cmds = [cmd for cmd in cmds if len(cmd) > 0]
        for line in cmds:
            le_args = dict()
            for pair in line:
                key, val = pair.strip()[2:].split("=")
                le_args[key] = val
            # if "xpid" not in le_args:
            #     le_args["xpid"] = generate_slug()

            all_args.append(le_args)

    executor.update_parameters(
        # examples setup
        partition=FLAGS.partition,
        # partition="prioritylab",
        comment="Neurips 2023 submission",
        time=1 * 72 * 60,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name=FLAGS.name,
        mem="512GB",
        cpus_per_task=10,
        gpus_per_node=1,
        constraint="volta32gb",
        array_parallelism=480,
    )

    print("\nAbout to submit", len(all_args), "jobs")

    if not FLAGS.debug:
        job = executor.map_array(LaunchExperiments(FLAGS.algo), all_args)

        for j in job:
            print("Submitted with job id: ", j.job_id)
            print(f"stdout -> {submitit_dir}/{j.job_id}_0_log.out")
            print(f"stderr -> {submitit_dir}/{j.job_id}_0_log.err")

        print(f"Submitted {len(job)} jobs!")

        print()
        print(submitit_dir)


if __name__ == "__main__":
    app.run(main)

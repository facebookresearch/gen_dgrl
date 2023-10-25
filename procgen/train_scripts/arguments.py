# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

parser = argparse.ArgumentParser(description="Make commands")

parser.add_argument(
    "--action",
    type=str,
    choices=["PRINT", "SAVE", "SLURM"],
    default="PRINT",
    help="PRINT: Print generated python commands out to terminal. "
    + "SAVE: Save generated python commands into a file. "
    + "SLURM: Schedule slurm jobs wtih generated commands.",
)

# Principle arguments
parser.add_argument(
    "--base_config",
    type=str,
    choices=["offline", "online"],
    default="offline",
    help="Base config where parameters are set with default values, and may be replaced by sweeping.",
)
parser.add_argument(
    "--grid_config",
    type=str,
    choices=["bc", "ppo", "bcq", "cql", "iql", "dt", "bct"],
    help="Name of the .json config for hyperparameter search-grid.",
)
parser.add_argument(
    "--num_trials", type=int, default=1, help="Number of seeds to be used for each hyperparameter setting."
)
parser.add_argument(
    "--module_name",
    type=str,
    choices=["offline.train_offline_agent", "online.trainer"],
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

# Misc
parser.add_argument(
    "--new_line",
    action="store_true",
    help="If true, print generated commands with arguments separated by new line; "
    + "otherwise arguments will be separated by space.",
)

# wandb
parser.add_argument("--wandb_api_key", type=str, default=None, help="wandb api key")
parser.add_argument("--wandb_base_url", type=str, default=None, help="wandb base url")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity")
parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")

# Slurm
parser.add_argument("--job_name", default="anyslurm", help="Slurm job name.")
parser.add_argument(
    "--dry_run", default=False, help="If true, a dry run will be performed and NO slurm jobs will be scheduled."
)

import argparse

from utils.utils import str2bool

parser = argparse.ArgumentParser(description="Data Collection")

# Checkpoint arguments
checkpoints = parser.add_mutually_exclusive_group()
checkpoints.add_argument("--checkpoint_path", type=str, help="path to the model checkpoint")
checkpoints.add_argument(
    "--ratio",
    type=float,
    default=None,
    metavar="[0.0 - 1.0]",
    help="Ratio of the expected test return of a checkpoint compared to expert data.",
)
parser.add_argument(
    "--ratio_checkpoint_dir", type=str, help="Directory where search of targeted checkpoint will happen."
)
parser.add_argument(
    "--ratio_dataset_dir",
    type=str,
    default="dataset",
    help="Root directory name for collected data under each checkpoint.",
)

# Dataset arguments
parser.add_argument(
    "--capacity",
    type=int,
    default=int(1e3),
    help="Size of the table to store interactions of the agent with the environment.",
)
parser.add_argument(
    "--dataset_saving_dir",
    type=str,
    default="dataset_saving_dir",
    help="directory to save episodes for offline training.",
)
parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae lambda parameter")
parser.add_argument("--gamma", type=float, default=0.999, help="discount factor for rewards")
parser.add_argument("--hidden_size", type=int, default=256, help="state embedding dimension")
parser.add_argument("--no_cuda", type=str2bool, default=False, help="If true, disable CUDA.")
parser.add_argument(
    "--num_env_steps",
    type=int,
    default=1e6,
    help="number of environment steps for the agent to interact with the environment.",
)
parser.add_argument("--num_processes", type=int, default=64, help="how many training CPU processes to use")
parser.add_argument(
    "--save_dataset",
    type=str2bool,
    help="If true, save episodes as datasets for offline training when training behavior policy.",
)
parser.add_argument("--seed", type=int, default=0, help="random seed")

# Procgen arguments.
parser.add_argument(
    "--distribution_mode",
    default="easy",
    choices=["easy", "hard", "extreme", "memory", "exploration"],
    help="distribution of envs for procgen",
)
parser.add_argument("--env_name", type=str, default="coinrun", help="environment to train on")
parser.add_argument("--num_levels", type=int, default=200, help="number of Procgen levels to use for training")
parser.add_argument("--start_level", type=int, default=0, help="start level id for sampling Procgen levels")

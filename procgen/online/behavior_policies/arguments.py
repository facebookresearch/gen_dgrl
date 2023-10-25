# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/arguments.py

import argparse

from utils.utils import str2bool

parser = argparse.ArgumentParser(description="RL")


# Algorithm arguments.
parser.add_argument("--algo", default="ppo", choices=["ppo"], help="algorithm to use")
parser.add_argument("--alpha", type=float, default=0.99, help="RMSprop optimizer apha")
parser.add_argument("--clip_param", type=float, default=0.2, help="ppo clip parameter")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy term coefficient")
parser.add_argument("--eps", type=float, default=1e-5, help="RMSprop optimizer epsilon")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae lambda parameter")
parser.add_argument("--gamma", type=float, default=0.999, help="discount factor for rewards")
parser.add_argument("--hidden_size", type=int, default=256, help="state embedding dimension")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max norm of gradients)")
parser.add_argument("--num_mini_batch", type=int, default=8, help="number of batches for ppo")
parser.add_argument("--num_steps", type=int, default=256, help="number of forward steps in A2C")
parser.add_argument("--ppo_epoch", type=int, default=1, help="number of ppo epochs")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--value_loss_coef", type=float, default=0.5, help="value loss coefficient (default: 0.5)")

# Procgen arguments.
parser.add_argument("--distribution_mode", default="easy", help="distribution of envs for procgen")
parser.add_argument("--env_name", type=str, default="coinrun", help="environment to train on")
parser.add_argument("--num_levels", type=int, default=200, help="number of Procgen levels to use for training")
parser.add_argument("--num_processes", type=int, default=64, help="how many training CPU processes to use")
parser.add_argument("--start_level", type=int, default=0, help="start level id for sampling Procgen levels")

# Training arguments
parser.add_argument("--archive_interval", type=int, default=50, help="number of updates after which model is saved.")
parser.add_argument("--checkpoint_path", type=str, default="", help="Directory to load model to start training.")
parser.add_argument("--log_interval", type=int, default=10, help="log interval, one log per n updates")
parser.add_argument(
    "--log_wandb",
    type=str2bool,
    default=True,
    help="If true, log of parameters and gradients to wandb",
)
parser.add_argument("--no_cuda", type=str2bool, default=False, help="If true, use CPU only.")
parser.add_argument("--num_env_steps", type=int, default=25e6, help="number of environment steps to train")
parser.add_argument("--model_saving_dir", type=str, default="models", help="Directory to save model during training.")
parser.add_argument(
    "--resume", type=str2bool, default=False, help="If true, load existing checkpoint to start training."
)
parser.add_argument("--wandb_base_url", type=str, default=None, help="wandb base url")
parser.add_argument("--wandb_api_key", type=str, default=None, help="wandb api key")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity")
parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
parser.add_argument("--xpid", type=str, default="debug", help="xpid name")

# Dataset arguments
parser.add_argument(
    "--dataset_saving_dir",
    type=str,
    default="dataset_saving_dir",
    help="directory to save episodes for offline training.",
)
parser.add_argument(
    "--save_dataset",
    type=str2bool,
    default=False,
    help="If true, save episodes as datasets when training behavior policy for offline training.",
)

"""
Main file which instantiates the Behavioral Cloning agent and trains it on the given dataset.
The dataset is loaded in the dataloader.
"""

import argparse
import logging
import os
import time

import procgen
import torch
import torch.nn as nn
from baselines.common.vec_env import VecExtractDictObs
from torch.utils.data import DataLoader

import wandb
from offline.agents import _create_agent
from offline.arguments import parser
from offline.dataloader import OfflineDataset, OfflineDTDataset
from offline.test_offline_agent import eval_agent, eval_DT_agent
from utils.filewriter import FileWriter
from utils.utils import set_seed
from utils.early_stopper import EarlyStop

args = parser.parse_args()
print(args)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

set_seed(args.seed)

if args.xpid is None:
    args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

log_dir = os.path.expandvars(os.path.expanduser(os.path.join(args.save_path, args.env_name)))
# check if final_model.pt already exists in the log_dir
if not os.path.exists(os.path.join(log_dir, args.xpid, "final_model.pt")):
    raise FileNotFoundError("Final model does not exist in the log_dir")

if os.path.exists(os.path.join(log_dir, args.xpid, "evaluate.csv")):
    # exit if final_model.pt already exists
    print("Final evaluate csv already exists in the log_dir")
    exit(0)
    
# Load dataset
extra_config = None
if args.algo in ["dt", "bct"]:
    dataset = OfflineDTDataset(
        capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile, context_len=args.dt_context_length, rtg_noise_prob=args.dt_rtg_noise_prob
    )
    extra_config = {"train_data_vocab_size": dataset.vocab_size, "train_data_block_size": dataset._block_size, "max_timesteps": max(dataset._timesteps), "dataset_size": len(dataset)}
    eval_max_return = dataset.get_max_return(multiplier=args.dt_eval_ret)


# create Procgen env
env = procgen.ProcgenEnv(num_envs=1, env_name=args.env_name)
env = VecExtractDictObs(env, "rgb")


# Initialize agent
agent = _create_agent(args, env=env, extra_config=extra_config)
agent.set_device(device)
print("Model Created!")

# load checkpoint and resume if resume flag is true
curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
print(f"Checkpoint Loaded!")


if args.algo in ["dt", "bct"]:
    test_mean_perf = eval_DT_agent(agent, eval_max_return, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode, num_episodes=100)
    train_mean_perf = eval_DT_agent(
        agent, eval_max_return, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode, num_episodes=100
    )
    val_mean_perf = eval_DT_agent(
        agent, eval_max_return, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode, num_episodes=100
    )
else:
    test_mean_perf = eval_agent(agent, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps, num_episodes=100)
    train_mean_perf = eval_agent(
        agent, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps, num_episodes=100
    )
    val_mean_perf = eval_agent(
        agent, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode, num_episodes=100
    )

# save dict to csv in logdir
with open(os.path.join(log_dir, args.xpid, "evaluate.csv"), "w") as f:
    f.write("final_test_ret,final_train_ret,final_val_ret\n")
    f.write(f"{test_mean_perf},{train_mean_perf},{val_mean_perf}\n")
    
print(f"Final Test Return: {test_mean_perf}")
print(f"Final Train Return: {train_mean_perf}")
print(f"Final Val Return: {val_mean_perf}")

print("Done!")
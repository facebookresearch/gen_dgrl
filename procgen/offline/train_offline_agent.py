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

# Setup wandb and offline logging
with open("wandb_info.txt") as file:
    lines = [line.rstrip() for line in file]
    os.environ["WANDB_BASE_URL"] = lines[0]
    os.environ["WANDB_API_KEY"] = lines[1]
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb_group = args.xpid[:-2][:126]  # '-'.join(args.xpid.split('-')[:-2])[:120]
    wandb_project = "OfflineRLBenchmark"
    wandb.init(project=wandb_project, entity=lines[2], config=args, name=args.xpid, group=wandb_group, tags=[args.algo,"final","seed_1","subop","8_june"])

log_dir = os.path.expandvars(os.path.expanduser(os.path.join(args.save_path, args.env_name)))
# check if final_model.pt already exists in the log_dir
if os.path.exists(os.path.join(log_dir, args.xpid, "final_model.pt")):
    # exit if final_model.pt already exists
    print("Final model already exists in the log_dir")
    exit(0)
filewriter = FileWriter(xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir)


def log_stats(stats):
    filewriter.log(stats)
    wandb.log(stats)


# logging.getLogger().setLevel(logging.INFO)

# Load dataset
pin_dataloader_memory = True
extra_config = None
if args.algo in ["dt", "bct"]:
    dataset = OfflineDTDataset(
        capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile, context_len=args.dt_context_length, rtg_noise_prob=args.dt_rtg_noise_prob
    )
    pin_dataloader_memory = True
    extra_config = {"train_data_vocab_size": dataset.vocab_size, "train_data_block_size": dataset._block_size, "max_timesteps": max(dataset._timesteps), "dataset_size": len(dataset)}
    eval_max_return = dataset.get_max_return(multiplier=args.dt_eval_ret)
    print("[DEBUG] Setting max eval return to ", eval_max_return)
else:
    dataset = OfflineDataset(
        capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile
    )
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_dataloader_memory) #, num_workers=8)

print("Dataset Loaded!")

# create Procgen env
env = procgen.ProcgenEnv(num_envs=1, env_name=args.env_name)
env = VecExtractDictObs(env, "rgb")

curr_epochs = 0
last_logged_update_count_at_restart = -1

# Initialize agent
agent = _create_agent(args, env=env, extra_config=extra_config)
agent.set_device(device)
print("Model Created!")

# wandb watch
# wandb.watch(agent.model_actor, log_freq=100)
# wandb.watch(agent.actor_dist, log_freq=100)
# wandb.watch(agent.model_v, log_freq=100)
# wandb.watch(agent.model_q1, log_freq=100)
# wandb.watch(agent.model_q2, log_freq=100)

# load checkpoint and resume if resume flag is true
if args.resume and os.path.exists(os.path.join(args.save_path, args.env_name, args.xpid, "model.pt")):
    curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
    last_logged_update_count_at_restart = filewriter.latest_update_count()
    print(f"Resuming checkpoint from Epoch {curr_epochs}, logged update count {last_logged_update_count_at_restart}")  
elif args.resume and os.path.exists(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt")):
    curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
    last_logged_update_count_at_restart = filewriter.latest_update_count()
    print(f"Resuming checkpoint from Epoch {curr_epochs}, logged update count {last_logged_update_count_at_restart}")
else:
    print("Starting from scratch!")

if args.early_stop:
    early_stopper = EarlyStop(wait_epochs=10, min_delta=0.1)

# Train agent
for epoch in range(curr_epochs, args.epochs):
    agent.train()
    epoch_loss = 0
    epoch_start_time = time.time()
    if args.algo in ["dt", "bct"]:
        for observations, actions, rtgs, timesteps, padding_mask in dataloader:
            observations, actions, rtgs, timesteps, padding_mask = (
                observations.to(device),
                actions.to(device),
                rtgs.to(device),
                timesteps.to(device),
                padding_mask.to(device)
            )
            stats_dict = agent.train_step(observations.float(), actions.long(), rtgs.float(), timesteps.long(), padding_mask.float())
            epoch_loss += stats_dict["loss"]
    else:
        for observations, actions, rewards, next_observations, dones in dataloader:
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(dim=1)
            if len(rewards.shape) == 1:
                rewards = rewards.unsqueeze(dim=1)
            if len(dones.shape) == 1:
                dones = dones.unsqueeze(dim=1)
            observations, actions, rewards, next_observations, dones = (
                observations.to(device),
                actions.to(device),
                rewards.to(device),
                next_observations.to(device),
                dones.to(device),
            )
            stats_dict = agent.train_step(
                observations.float(), actions.long(), rewards.float(), next_observations.float(), dones.float()
            )
            epoch_loss += stats_dict["loss"]
    epoch_end_time = time.time()

    # evaluate the agent on procgen environment
    if epoch % args.eval_freq == 0:
        inf_start_time = time.time()
        if args.algo in ["dt", "bct"]:
            test_mean_perf = eval_DT_agent(
                agent,
                eval_max_return,
                device,
                env_name=args.env_name,
                start_level=args.num_levels+50,
                distribution_mode=args.distribution_mode
            )
            val_mean_perf = eval_DT_agent(
                agent,
                eval_max_return,
                device,
                env_name=args.env_name,
                num_levels=50,
                start_level=args.num_levels,
                distribution_mode=args.distribution_mode
            )
            train_mean_perf = eval_DT_agent(
                agent,
                eval_max_return,
                device,
                env_name=args.env_name,
                num_levels=args.num_levels,
                start_level=0,
                distribution_mode=args.distribution_mode
            )
        else:
            test_mean_perf = eval_agent(
                agent,
                device,
                env_name=args.env_name,
                start_level=args.num_levels+50,
                distribution_mode=args.distribution_mode,
                eval_eps=args.eval_eps,
            )
            val_mean_perf = eval_agent(
                agent,
                device,
                env_name=args.env_name,
                num_levels=50,
                start_level=args.num_levels,
                distribution_mode=args.distribution_mode,
                eval_eps=args.eval_eps,
            )
            train_mean_perf = eval_agent(
                agent,
                device,
                env_name=args.env_name,
                num_levels=args.num_levels,
                start_level=0,
                distribution_mode=args.distribution_mode,
                eval_eps=args.eval_eps,
            )
        inf_end_time = time.time()

        print(
            f"Epoch: {epoch + 1} | Loss: {epoch_loss / len(dataloader)} | Time: {epoch_end_time - epoch_start_time} \
                | Train Returns (mean): {train_mean_perf} | Validation Returns (mean): {val_mean_perf} | Test Returns (mean): {test_mean_perf}"
        )

        print(epoch+1)
        if (epoch+1) > last_logged_update_count_at_restart:
            stats_dict.update(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss / len(dataloader),
                    "epoch_time": epoch_end_time - epoch_start_time,
                    "inf_time": inf_end_time - inf_start_time,
                    "train_rets_mean": train_mean_perf,
                    "test_rets_mean": test_mean_perf,
                    "val_rets_mean": val_mean_perf,
                }
            )
            log_stats(stats_dict)

            # Save agent and number of epochs
            if args.resume and (epoch+1) % args.ckpt_freq == 0:
                curr_epochs = epoch + 1
                agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
                agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, f"model_{epoch}.pt"))
                
        if args.early_stop:
            if early_stopper.should_stop(epoch, val_mean_perf):
                print("[DEBUG]: Early stopping")             
                break

if args.algo in ["dt", "bct"]:
    test_mean_perf = eval_DT_agent(agent, eval_max_return, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode)
    train_mean_perf = eval_DT_agent(
        agent, eval_max_return, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode
    )
    val_mean_perf = eval_DT_agent(
        agent, eval_max_return, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode
    )
else:
    test_mean_perf = eval_agent(agent, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps)
    train_mean_perf = eval_agent(
        agent, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps
    )
    val_mean_perf = eval_agent(
        agent, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode
    )
wandb.log({"final_test_ret": test_mean_perf, "final_train_ret": train_mean_perf, "final_val_ret": val_mean_perf}, step=(epoch + 1))
filewriter.log_final_test_eval({
        'final_test_ret': test_mean_perf,
        'final_train_ret': train_mean_perf,
        'final_val_ret': val_mean_perf
    })
if args.resume:
    agent.save(num_epochs=args.epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))

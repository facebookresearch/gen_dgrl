# Run PPO final checkpoints on Procgen test levels
import argparse
import os
import random

import numpy as np
import torch

from online.behavior_policies.envs import make_venv
from online.behavior_policies.model import PPOnet
from utils.utils import LogItemType, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--xpid", type=str, default="debug")


def evaluate_one_game(env_name: str, num_episodes=5, seed=1, start_level=0, num_levels=0):
    # Initialize model
    model = PPOnet((3, 64, 64), 15, base_kwargs={"hidden_size": 256})
    # Load checkpoint
    # SEED = 1
    model_path = "YOUR_MODEL_PATH"
    model_path = [f.path for f in os.scandir(model_path)][0]
    print(f"Loading checkpoint from {model_path} ...")
    try:
        checkpoint_states = torch.load(model_path)
        model.load_state_dict(checkpoint_states[LogItemType.MODEL_STATE_DICT.value])
    except Exception:
        print(f"Unable to load checkpoint from {model_path}, model is initialized randomly.")
    device = "cuda:0"
    model.to(device)
    # Initialize Env

    test_envs = make_venv(
        num_envs=1,
        env_name=env_name,
        device=device,
        **{
            "num_levels": num_levels,
            "start_level": start_level,
            "distribution_mode": "easy",
            "ret_normalization": False,
            "obs_normalization": True,
        },
    )
    # # Roll out
    # model.eval()
    # rewards_per_level = []
    # obs = test_envs.reset()
    # while len(rewards_per_level) < num_episodes:
    #     with torch.no_grad():
    #         _, action, _ = model.act(obs)
    #     obs, _reward, _done, infos = test_envs.step(action)
    #     for info in infos:
    #         if "episode" in info.keys():
    #             rewards_per_level.append(info["episode"]["r"])
    # # print(f"LEVEL: {level} - AVERAGE REWARDS OVER {num_episodes} EPISODES: {np.mean(rewards_per_level)}")
    # rewards_over_all_levels.append(np.mean(rewards_per_level))
    
    eval_episode_rewards = []
    model.eval()
    for _ in range(num_episodes):
        obs = test_envs.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                _, action, _ = model.act(obs)
            obs, reward, done, _ = test_envs.step(action)
            episode_reward += reward.item()
        eval_episode_rewards.append(episode_reward)
    return eval_episode_rewards

from procgen.env import ENV_NAMES

# random.seed(1337)

args = parser.parse_args()
set_seed(args.seed)
# Change test levels based on the num_levels that PPO is trained!!!
# levels = random.sample(range(100050, 10_000_000), 10)
# print(levels)
train_result = {}
for env_name in ENV_NAMES:
    # env_name = "plunder"
    rewards = evaluate_one_game(env_name, num_episodes=10, seed=args.seed, start_level=40, num_levels=1)
    print(f"ENV: {env_name} - REWARDS OVER LEVELS: {np.mean(rewards)}")
    train_result[env_name] = np.mean(rewards)
print(train_result)

# test_result = {}
# for env_name in ENV_NAMES:
#     # env_name = "plunder"
#     rewards = evaluate_one_game(env_name, num_episodes=100, seed=args.seed, start_level=250, num_levels=0)
#     print(f"ENV: {env_name} - REWARDS OVER LEVELS: {np.mean(rewards)}")
#     test_result[env_name] = np.mean(rewards)
# print(test_result)


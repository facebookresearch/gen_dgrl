# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import procgen
import torch
import torch.nn as nn
from baselines.common.vec_env import VecExtractDictObs


def eval_agent(
    agent: nn.Module,
    device,
    env_name="miner",
    num_levels=0,
    start_level=0,
    distribution_mode="easy",
    eval_eps=0.001,
    num_episodes=10,
):
    # Sample Levels From the Full Distribution
    env = procgen.ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    eval_episode_rewards = []
    agent.eval()
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if obs.shape[1] != 3:
                obs = obs.transpose(0, 3, 1, 2)
            obs = torch.from_numpy(obs).float().to(device)
            # normalize obs to [0, 1] if [0,255]
            # if obs.max() > 1.0:
            #     obs /= 255.0
            action = agent.eval_step(obs, eps=eval_eps)
            # using numpy, if action is of shape [1,1] then convert it to [1]
            if action.shape == (1, 1):
                action = action[0]
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        eval_episode_rewards.append(episode_reward)
    mean_eval_episode_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)
    return mean_eval_episode_reward


def eval_DT_agent(
    agent: nn.Module,
    ret,
    device,
    env_name="miner",
    num_levels=0,
    start_level=0,
    distribution_mode="easy",
    num_episodes=10,
):
    # Sample Levels From the Full Distribution
    env = procgen.ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    eval_episode_rewards = []
    info_episode_rewards = []
    agent.eval()
    done = True
    for _ in range(num_episodes):
        state = env.reset()
        # done = False

        if state.shape[1] != 3:
            state = state.transpose(0, 3, 1, 2)
        state = torch.from_numpy(state).type(torch.float32).to(device).unsqueeze(0)
        # normalize state to [0, 1] if [0,255]
        if state.max() > 1.0:
            state /= 255.0

        rtgs = [ret]
        # first state is from env, first rtg is target return, and first timestep is 0
        sampled_action = agent.sample(
            state,
            1,
            temperature=1.0,
            sample=True,
            actions=None,
            rtgs=torch.tensor(rtgs, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(-1),
            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
        )
        j = 0
        all_states = state
        actions = []

        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            action = sampled_action[0]
            actions += [sampled_action]
            if action.shape == (1, 1):
                action = action[0]
            state, reward, done, infos = env.step(action.cpu().numpy())

            if state.shape[1] != 3:
                state = state.transpose(0, 3, 1, 2)
            state = torch.from_numpy(state).type(torch.float32).to(device)
            # normalize state to [0, 1] if [0,255]
            if state.max() > 1.0:
                state /= 255.0
            reward_sum += reward[0]
            j += 1

            for info in infos:
                if "episode" in info.keys():
                    info_episode_rewards.append(info["episode"]["r"])

            if done:
                eval_episode_rewards.append(reward_sum)
                break

            state = state.unsqueeze(0).to(device)

            all_states = torch.cat([all_states, state], dim=0)

            if reward.shape != (1, 1):
                reward = torch.from_numpy(reward).type(torch.float32).unsqueeze(-1)
            rtgs += [rtgs[-1] - reward]
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            # st()
            sampled_action = agent.sample(
                all_states.unsqueeze(0),
                1,
                temperature=1.0,
                sample=True,
                actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0),
                rtgs=torch.tensor(rtgs, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(-1),
                timesteps=(min(j, agent.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)),
            )

    mean_eval_episode_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)
    # print(mean_eval_episode_reward, info_episode_rewards)
    return mean_eval_episode_reward

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import procgen
import torch
from baselines.common.vec_env import VecEnvWrapper, VecExtractDictObs, VecMonitor, VecNormalize
from gym.spaces.box import Box
from procgen import ProcgenEnv


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device, normalize=True):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device
        self.normalize = normalize

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype,
        )

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)

        obs = torch.from_numpy(obs).float().to(self.device)
        if self.normalize:
            obs /= 255.0

        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)

        obs = torch.from_numpy(obs).float().to(self.device)
        if self.normalize:
            obs /= 255.0

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


def make_venv(num_envs, env_name, device, **kwargs):
    """
    Function to create a vectorized environment.
    """
    if env_name in procgen.env.ENV_NAMES:
        num_levels = kwargs.get("num_levels", 1)
        start_level = kwargs.get("start_level", 0)
        distribution_mode = kwargs.get("distribution_mode", "easy")
        paint_vel_info = kwargs.get("paint_vel_info", False)
        use_sequential_levels = kwargs.get("use_sequential_levels", False)
        ret_normalization = kwargs.get("ret_normalization", False)
        obs_normalization = kwargs.get("obs_normalization", False)

        venv = ProcgenEnv(
            num_envs=num_envs,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            paint_vel_info=paint_vel_info,
            use_sequential_levels=use_sequential_levels,
        )
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False, ret=ret_normalization)

        envs = VecPyTorchProcgen(venv, device, normalize=obs_normalization)
    else:
        raise ValueError(f"Unsupported env {env_name}")

    return envs

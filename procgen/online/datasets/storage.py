import collections
import datetime
import io
import os
import tempfile
from typing import Dict

import numpy as np
import torch

from utils.utils import DatasetItemType


def save_episode(episode: Dict[str, np.ndarray], path: str):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


class RolloutStorage(object):
    def __init__(
        self,
        observation_space,
        action_space,
        n_envs: int = 1,
        capacity: int = int(1e6),
        save_episode: bool = False,
        storage_path: str = None,
    ) -> None:
        """

        Args:
            observation_space (gym.spaces.Box): the observation space for the environment
            action_space (gym.spaces.Discrete): the action space for the environment
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.capacity = capacity

        self.n_episodes = 0
        self.save_episode = save_episode
        self.storage_path = storage_path
        self.set_storage_path(storage_path)

        self.setup()

    def setup(self) -> None:
        """
        Initializing buffers and indexes
        """
        # Buffers for each element
        self._obs_buffer = torch.zeros(size=(self.capacity + 1, self.n_envs, *self.observation_space))
        self._reward_buffer = torch.zeros(size=(self.capacity, self.n_envs, 1))
        self._done_buffer = torch.empty(size=(self.capacity, self.n_envs), dtype=torch.bool)

        action_shape = 1 if self._is_discrete() else self.action_space.shape[0]
        self._action_buffer = torch.zeros(size=(self.capacity, self.n_envs, action_shape))
        if self._is_discrete():
            self._action_buffer = self._action_buffer.long()

        self._level_seeds_buffer = np.zeros(shape=(self.n_envs,), dtype=np.int32)

        # Index
        self._idx = 0
        self._prev_idx = -1

    def reset(self) -> None:
        """
        Reset buffers and indexes
        """
        # Buffers for each element
        self._obs_buffer.zero_()
        self._reward_buffer.zero_()
        self._done_buffer.zero_()
        self._action_buffer.zero_()
        self._level_seeds_buffer = np.zeros(shape=(self.n_envs,), dtype=np.int32)

        # Index
        self._idx = 0
        self._prev_idx = -1

    def set_storage_path(self, storage_path) -> None:
        if not self.save_episode:
            return

        self.storage_path = storage_path
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)

        if storage_path is None:
            self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")

    def insert_initial_observations(self, init_obs) -> None:
        """
        Add only the first observations obtained when reseting the environments.

        Example:
            >>> venv = ProcgenEnv(num_envs=3, env_name="miner", num_levels=200, start_level=0, distribution_mode="easy")
            >>> venv = VecExtractDictObs(venv, "rgb")
            >>> obs = venv.reset()
            >>> buffer = ReplayBuffer(
                    observation_space=venv.observation_space,
                    action_space=venv.action_space,
                    n_envs=venv.num_envs,
                    capacity=4,
                    save_episode=False,
                )
            >>> buffer.setup()

            >>> buffer.insert_initial_observations(obs)

        Args:
            init_obs (np.ndarray): initial observations
        """
        self._obs_buffer[self._idx].copy_(init_obs)

        if self.save_episode:
            if not hasattr(self, "ongoing_episodes"):
                self.ongoing_episodes = [None] * self.n_envs

            for env_idx in range(self.n_envs):
                self.ongoing_episodes[env_idx] = collections.defaultdict(list)
                self.ongoing_episodes[env_idx][DatasetItemType.OBSERVATIONS.value].append(
                    init_obs[env_idx].detach().cpu().numpy()
                )  # init_obs: (n_envs, 3, 64, 64)

    def to(self, device):
        """
        Move tensors in the buffers to a CPU / GPU device.
        """
        self._obs_buffer = self._obs_buffer.to(device)
        self._reward_buffer = self._reward_buffer.to(device)
        self._done_buffer = self._done_buffer.to(device)
        self._action_buffer = self._action_buffer.to(device)

    def insert(self, obs, actions, rewards, dones, infos=[]) -> None:
        r"""
        Insert tuple of (observations, actions, rewards, dones) into corresponding buffers.
        An 's' at the end of each parameter indicates that the environments can be vectorized.

        Example:
            >>> venv = ProcgenEnv(num_envs=3, env_name="miner", num_levels=200, start_level=0, distribution_mode="easy")
            >>> venv = VecExtractDictObs(venv, "rgb")
            >>> obs = venv.reset()

            >>> action = venv.action_space.sample()
            >>> actions = np.array([action] * venv.num_envs)
            >>> obs, rewards, dones, _infos = venv.step(actions)

            >>> buffer.insert(obs, actions, rewards, dones)

        Args:
            obs: observations or states observed from the environments after the agents perform the actions.
            actions: The actions sampled from a certain policy or randomly for the agents to perform.
            rewards: The immidiate rewards that the agents received from the environments after performing the actions.
            dones: A boolean vector whose elements indicate whether the episode terminates in an environment.
        """
        # Update buffers
        if len(rewards.shape) == 3:
            rewards = rewards.squeeze(2)

        self._obs_buffer[self._idx + 1].copy_(obs)
        self._action_buffer[self._idx].copy_(actions)
        self._reward_buffer[self._idx].copy_(rewards)
        self._done_buffer[self._idx] = torch.tensor(dones)
        # When done == True (episode is completed), 'level_seed' in info will be the new seed,
        # and 'prev_level_seed' will be the seed that was used in that completed episode.
        # Save 'prev_level_seed' before saving an completed episode.
        self._level_seeds_buffer = np.array([info.get("prev_level_seed", -1) for info in infos])

        # Update index
        self._prev_idx = self._idx
        self._idx = (self._idx + 1) % self.capacity

        # Keep track of current episodes and save completed ones.
        if self.save_episode:
            for env_idx in range(self.n_envs):
                self.ongoing_episodes[env_idx][DatasetItemType.OBSERVATIONS.value].append(
                    obs[env_idx].detach().cpu().numpy()
                )
                self.ongoing_episodes[env_idx][DatasetItemType.ACTIONS.value].append(
                    actions[env_idx].detach().cpu().numpy()
                )
                self.ongoing_episodes[env_idx][DatasetItemType.REWARDS.value].append(
                    rewards[env_idx].detach().cpu().numpy()
                )
                self.ongoing_episodes[env_idx][DatasetItemType.DONES.value].append(dones[env_idx])

            if any(dones):
                done_env_idxs = np.where(dones == 1)[0]
                self._save_terminated_episodes(done_env_idxs)
                self._reset_terminated_episodes(done_env_idxs)

        # Update with current level seed after saving completed episodes.
        self._level_seeds_buffer = np.array([info.get("level_seed", -1) for info in infos])

    def save(self) -> None:
        """
        Save the replay buffer
        """
        pass

    def _is_discrete(self) -> bool:
        """
        Determine if the environment action space is discrete or continuous
        """
        return self.action_space.__class__.__name__ == "Discrete"

    def _save_terminated_episodes(self, done_env_idxs: np.ndarray) -> None:
        """
        Save all terminated episodes among the n_envs environments.

        Args:
            done_env_idxs (np.ndarray): indexs of environments having episode terminated at the current step.
        """
        for env_idx in done_env_idxs:
            self._save_episode(env_idx)

    def _save_episode(self, env_idx: int) -> None:
        """
        Save a single episode into file.

        Args:
            env_idx (int): the index of the environment, ranging [0, n_envs-1] inclusive.
        """
        # Convert list to numpy array
        episode = {}
        for k, v in self.ongoing_episodes[env_idx].items():
            first_value = v[0]
            if isinstance(first_value, np.ndarray):
                dtype = first_value.dtype
            elif isinstance(first_value, int):
                dtype = np.int64
            elif isinstance(first_value, float):
                dtype = np.float32
            elif isinstance(first_value, bool):
                dtype = np.bool_
            episode[k] = np.array(v, dtype=dtype)

        # File name
        episode_idx = self.n_episodes
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        episode_len = len(self.ongoing_episodes[env_idx]["rewards"])
        level_seed = self._level_seeds_buffer[env_idx]
        total_rewards = np.squeeze(sum(self.ongoing_episodes[env_idx]["rewards"]))
        episode_filename = f"{timestamp}_{episode_idx}_{episode_len}_{level_seed}_{total_rewards:.2f}.npz"

        # Store the episode
        self.n_episodes += 1
        save_episode(episode, os.path.join(self.storage_path, episode_filename))

    def _reset_terminated_episodes(self, done_env_idxs: np.ndarray) -> None:
        """
        Reset references of all terminated episodes of the n_envs environments.

        Args:
            done_env_idxs (np.ndarray): indexs of environments having episode terminated at the current step.
        """
        for env_idx in done_env_idxs:
            self._reset_terminated_episode(env_idx)

    def _reset_terminated_episode(self, env_idx: int) -> None:
        """
        Reset the reference of a single terminated episode.

        Args:
            env_idx (int): the index of the environment, ranging [0, n_envs-1] inclusive.
        """
        # clear the reference
        self.ongoing_episodes[env_idx] = collections.defaultdict(list)

        # the next_obs of the previous (saved) terminated episode is the init_obs of the next episode.
        # self._prev_idx has range [1, capacity+1] inclusive, covers the roll over edge cases.
        self.ongoing_episodes[env_idx][DatasetItemType.OBSERVATIONS.value].append(
            self._obs_buffer[self._prev_idx + 1][env_idx].detach().cpu().numpy()
        )

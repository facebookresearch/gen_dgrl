import collections
import datetime
import io
import os
import tempfile
from typing import Dict

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def save_episode(episode: Dict[str, np.ndarray], directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with open(os.path.join(directory, filename), "wb") as f:
            f.write(bs.read())


class ReplayBuffer(object):
    """
    Create a replay buffer for procgen environment.

    Procgen Environment information: https://github.com/openai/procgen#environments
    """

    def __init__(
        self,
        observation_space,
        action_space,
        n_envs: int = 1,
        n_steps: int = 256,
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
        self.n_steps = n_steps

        self.n_episodes = 0
        self.save_episode = save_episode
        self.storage_path = storage_path
        if save_episode and storage_path is None:
            self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")

        self.setup()

    def setup(self) -> None:
        """
        Initializing buffers and indexes
        """
        # Buffers for each element
        self._obs_buffer = torch.zeros(size=(self.n_steps + 1, self.n_envs, *self.observation_space))
        self._reward_buffer = torch.zeros(size=(self.n_steps, self.n_envs, 1))
        self._done_buffer = torch.empty(size=(self.n_steps, self.n_envs), dtype=torch.bool)
        self.value_preds = torch.zeros(size=(self.n_steps + 1, self.n_envs, 1))
        self.returns = torch.zeros(size=(self.n_steps + 1, self.n_envs, 1))
        self._action_log_probs = torch.zeros(size=(self.n_steps, self.n_envs, 1))

        action_shape = 1 if self._is_discrete() else self.action_space.shape[0]
        self._action_buffer = torch.zeros(size=(self.n_steps, self.n_envs, action_shape))
        if self._is_discrete():
            self._action_buffer = self._action_buffer.long()

        self._masks = torch.ones(size=(self.n_steps + 1, self.n_envs, 1))

        # Index
        self._idx = 0
        self._prev_idx = -1

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
                    n_steps=4,
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
                self.ongoing_episodes[env_idx]["obs"].append(
                    np.array(init_obs[env_idx].detach().cpu())
                )  # init_obs: (n_envs, 3, 64, 64)

    def to(self, device):
        """
        Move tensors in the buffers to a CPU / GPU device.
        """
        self._obs_buffer = self._obs_buffer.to(device)
        self._reward_buffer = self._reward_buffer.to(device)
        self._done_buffer = self._done_buffer.to(device)
        self._action_buffer = self._action_buffer.to(device)
        self._action_log_probs = self._action_log_probs.to(device)
        self.returns = self.returns.to(device)
        self.value_preds = self.value_preds.to(device)
        self._masks = self._masks.to(device)

    def insert(self, obs, actions, rewards, dones, action_log_probs, value_preds, masks) -> None:
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
        if len(rewards.shape) == 3:
            rewards = rewards.squeeze(2)

        self._obs_buffer[self._idx + 1].copy_(obs)
        self._action_buffer[self._idx].copy_(actions)
        self._reward_buffer[self._idx].copy_(rewards)
        self._done_buffer[self._idx] = torch.tensor(dones)
        self._action_log_probs[self._idx].copy_(action_log_probs)
        self.value_preds[self._idx].copy_(value_preds)
        self._masks[self._idx + 1].copy_(masks)

        # Update index
        self._prev_idx = self._idx
        self._idx = (self._idx + 1) % self.n_steps

        if self.save_episode:
            for env_idx in range(self.n_envs):
                self.ongoing_episodes[env_idx]["obs"].append(np.array(obs[env_idx].detach().cpu()))
                self.ongoing_episodes[env_idx]["actions"].append(np.array(actions[env_idx].detach().cpu()))
                self.ongoing_episodes[env_idx]["rewards"].append(np.array(rewards[env_idx].detach().cpu()))
                self.ongoing_episodes[env_idx]["dones"].append(dones[env_idx])

            if any(dones):
                done_env_idxs = np.where(dones == 1)[0]
                self._save_terminated_episodes(done_env_idxs)
                self._reset_terminated_episodes(done_env_idxs)

    def after_update(self):
        self._obs_buffer[0].copy_(self._obs_buffer[-1])
        self._masks[0].copy_(self._masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self._reward_buffer.shape[0])):
            delta = (
                self._reward_buffer[step]
                + gamma * self.value_preds[step + 1] * self._masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * gae_lambda * self._masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        n_steps, n_envs = self._reward_buffer.shape[0:2]
        batch_size = n_envs * n_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({n_envs}) ",
                f"* number of steps ({n_steps}) = {n_envs * n_steps} ",
                f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).",
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = self._obs_buffer[:-1].view(-1, *self._obs_buffer.shape[2:])[indices]
            actions_batch = self._action_buffer.view(-1, self._action_buffer.shape[-1])[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self._action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ

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
        episode_idx = self.n_episodes
        episode_len = len(self.ongoing_episodes[env_idx]["rewards"])
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

        # Store the episode
        self.n_episodes += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        episode_filename = f"{timestamp}_{episode_idx}_{episode_len}.npz"
        save_episode(episode, self.storage_path, episode_filename)

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
        # self._prev_idx has range [1, n_steps+1] inclusive, covers the roll over edge cases.
        self.ongoing_episodes[env_idx]["obs"].append(
            np.array(self._obs_buffer[self._prev_idx + 1][env_idx].detach().cpu())
        )

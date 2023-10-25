# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import os
import gc
from itertools import accumulate
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Callable, Iterable, Tuple, Union
import numpy as np
import torch
import random

from utils.utils import DatasetItemType


def load_episode(path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        episode['observations'] = episode['observations'].astype(np.uint8)
        episode['rewards'] = episode['rewards'].astype(np.float)
        return episode


def compute_episode_length(episode: Dict[str, np.ndarray]) -> int:
    return len(episode[DatasetItemType.ACTIONS.value])


def fetch_return_from_path(name: Path) -> float:
    return float(name.stem.split("_")[-1])


class OfflineDataset(torch.utils.data.Dataset):
    """
    Load episodes from files and sample a batch (obs, next_obs, action, reward, done)
    from one of the loaded episodes.
    """

    _capacity: int
    _episodes_dir_path: List[str]
    _episodes: List[Dict[str, np.ndarray]]
    _loaded: bool
    _size: int
    _num_transitions: int
    _percentile: float
    _zero_out_last_obs: bool
    _episode_lengths: List[int]
    _capacity_type: str
    _specific_level_seed: Optional[int]
    _max_episodes: Optional[int]
    

    def __init__(
        self, capacity: int, episodes_dir_path: List[str], percentile: float = 1.0, zero_out_last_obs: bool = True,
        capacity_type: str = 'transitions', specific_level_seed: Optional[int] = None, max_episodes: Optional[int] = None
    ) -> None:
        self._capacity = capacity
        self._episodes_dir_path = (
            [directory for directory in episodes_dir_path]
            if isinstance(episodes_dir_path, list)
            else [episodes_dir_path]
        )
        self._episodes = []
        self._loaded = False
        self._size = 0
        self._percentile = percentile
        self._zero_out_last_obs = zero_out_last_obs
        self._specific_level_seed = specific_level_seed
        self._capacity_type = capacity_type
        if self._capacity_type == 'episodes':
            assert max_episodes is not None, "max_episodes must be specified when capacity_type is 'episodes'"
        elif self._capacity_type == 'transitions':
            assert max_episodes is None, "max_episodes must be None when capacity_type is 'transitions'"
        self._max_episodes = max_episodes
        self._sort_by_return_and_load_by_percentile()
        
    def _calc_average_return(self) -> float:
        # calculate average return across all episodes
        rewards = []
        for episode in self._episodes:
            rewards.append(np.sum(episode[DatasetItemType.REWARDS.value]))
        return np.mean(rewards)
            

    def _sort_by_return_and_load_by_percentile(self) -> None:
        if self._loaded is True:
            return

        episode_filenames = sorted(
            [f.path for directory in self._episodes_dir_path for f in os.scandir(directory)],
            key=lambda path: OfflineDataset._fetch_return_from_path(path),
            reverse=True,
        )
        print(f"[DEBUG] Total number of episodes: {len(episode_filenames)}.")

        num_transitions = int(self._capacity * self._percentile)
        print(
            f"[DEBUG] Capacity: {self._capacity}. "
            + f"Loading {num_transitions} ({100 * self._percentile}% of {self._capacity}) transitions ..."
        )

        # Store all episodes (capped by _capacity and _percentile) into _episodes
        for name in episode_filenames:
            if self._specific_level_seed is not None:
                level_seed = int(name.split('/')[-1].split('_')[-2])
                if level_seed != self._specific_level_seed:
                    continue
            episode = load_episode(name)
            curr_episode_len = compute_episode_length(episode)
            if curr_episode_len <= 0:
                # Filter out invalid episodes
                continue

            self._episodes.append(episode)
            self._size += curr_episode_len
            if self._capacity_type  == 'transitions' and self._size > num_transitions:
                break
            elif self._capacity_type == 'episodes' and len(self._episodes) >= self._max_episodes:
                break

        print(f"[DEBUG] Loaded {len(self._episodes)} episodes with {self._size} transitions in total!")
        self._loaded = True
        self._num_transitions = min(num_transitions, self._size)

        # _episode_lengths store the cumulated sum of lengths of episodes that are before the current one (included).
        # As if all stored episodes are concatenated.
        self._episode_lengths = list(accumulate(compute_episode_length(episode) for episode in self._episodes))

    @staticmethod
    def _fetch_return_from_path(path: str) -> float:
        """
        Example:
        >>> path = '/my_documents/model_1_3.5.pt'
        >>> score = fetch_return_from_path(path)
        >>> score
        3.5
        """
        return float(Path(path).stem.split("_")[-1])

    def __len__(self) -> int:
        return self._num_transitions

    def __getitem__(self, index):
        """
        Example:
            Suppose we have 3 episodes with length: 5, 3, 7. The capacity of the dataset is 10, percentile 100%.

            Then we will store those 3 episodes, and _episode_lengths == [5, 5+3, 5+3+7].

            The sample index will be 0 <= index <= 9, given the capacity as 10.

            If index == 6, we know index >= 5 and index < 8, thus it should be in the middle episode, and its offset
            to the beginning of that episode is (6-5) == 1.
        """
        # Use binary search to find out which episode whose transitions match the index.
        episode_idx = bisect.bisect_right(self._episode_lengths, index)
        curr = self._episodes[episode_idx]
        # Compute the offset within the episode
        offset_in_episode = index - (0 if episode_idx == 0 else self._episode_lengths[episode_idx - 1])

        # Fetch corresponding state-action tuple.
        obs = curr[DatasetItemType.OBSERVATIONS.value][offset_in_episode]
        next_obs = (
            # If the episode is completed, the next_obs is a zero vector
            np.zeros_like(obs)
            if self._zero_out_last_obs and curr[DatasetItemType.DONES.value][offset_in_episode]
            else curr[DatasetItemType.OBSERVATIONS.value][offset_in_episode + 1]
        )
        actions = curr[DatasetItemType.ACTIONS.value][offset_in_episode]
        rewards = curr[DatasetItemType.REWARDS.value][offset_in_episode]
        dones = curr[DatasetItemType.DONES.value][offset_in_episode] != 0

        return (obs, actions, rewards, next_obs, dones)


class OfflineDTDataset:
    """
    A dataset designed for Decision Transformer.
    """

    _capacity: int
    _episodes_dir_path: List[Path]
    _episodes: List[Dict[str, np.ndarray]]
    _loaded: bool
    _size: int
    _rtg_noise_prob: float
    _short_traj_count: int
    _context_len: int
    _percentile: float
    _specific_level_seed: Optional[int]
    _capacity_type: str

    def __init__(
        self,
        capacity: int,
        episodes_dir_path: List[str],
        context_len: int,
        rtg_noise_prob: float,
        percentile: float = 1.0,
        specific_level_seed: Optional[int] = None, 
        capacity_type: str = 'transitions'
    ) -> None:
        self._capacity = capacity
        self._episodes_dir_path = (
            [Path(directory) for directory in episodes_dir_path]
            if isinstance(episodes_dir_path, list)
            else [Path(episodes_dir_path)]
        )
        self.context_len = context_len
        self._block_size = context_len * 3
        self._rtg_noise_prob = rtg_noise_prob
        if self._rtg_noise_prob > 0:
            print("[DEBUG] Using RTG noise with probability", self._rtg_noise_prob)
        self._short_traj_count = 0

        self._episodes = []
        self._loaded = False
        self._size = 0
        self._percentile = percentile
        self._max_return = -float("inf")
        
        self._specific_level_seed = specific_level_seed
        self._capacity_type = capacity_type

        self._load()
        self._reassemble()

    def _load(self) -> None:
        if self._loaded is True:
            return

        episode_filenames = [
            name for gens in (directory.rglob("*.npz") for directory in self._episodes_dir_path) for name in gens
        ]
        
        
        # Randomly shuffle episode files
        random.shuffle(episode_filenames)
        
        print(f"[DEBUG] Total number of episodes: {len(episode_filenames)}.")
        print(f"[DEBUG] Capacity: {self._capacity}. Loading {self._capacity} {self._capacity_type} ...")

        # Initialize count of tuples
        tuples_count = 0

        # Target number of tuples
        target_tuples = self._capacity
        
        for name in tqdm(episode_filenames):
            if self._specific_level_seed is not None:
                level_seed = int(str(name).split('/')[-1].split('_')[-2])
                if level_seed != self._specific_level_seed:
                    continue
            # episode = load_episode(name)
            # curr_episode_len = compute_episode_length(episode)
            curr_episode_len = int(name.stem.split('_')[-3])
            if curr_episode_len <= 3:
                # Filter out invalid episodes
                continue

            self._episodes.append(name)
            tuples_count += curr_episode_len
            current_return = fetch_return_from_path(name)
            if tuples_count >= target_tuples:
                break
            self._max_return = max(self._max_return, current_return)

            # if self._size > self._capacity:
            #     break

        self._size = tuples_count
        print(
            f"[DEBUG] Loaded {len(self._episodes)} episodes with {self._size} transitions in total!"
            + f"Maximum return: {self._max_return}"
        )
        self._loaded = True

    def _reassemble(self) -> None:
        # Get shapes
        first = load_episode(self._episodes[0])
        first_obs = first[DatasetItemType.OBSERVATIONS.value][0]
        first_action = first[DatasetItemType.ACTIONS.value][0]
        first_reward = first[DatasetItemType.REWARDS.value][0]
        first_done = first[DatasetItemType.DONES.value][0]

        # Pick top percentile and reassemble
        self._obs = np.empty(shape=(self._size, *first_obs.shape), dtype=first_obs.dtype)
        self._actions = np.empty(shape=(self._size,), dtype=first_action.dtype)
        self._rewards = np.empty(shape=(self._size,), dtype=first_reward.dtype)
        self._return_to_gos = np.empty(shape=(self._size,), dtype=first_reward.dtype)
        self._dones = np.empty(shape=(self._size, *first_done.shape), dtype=np.bool_)
        self._timesteps = np.empty(shape=(self._size,), dtype=np.int)

        print(f"[DEBUG] Assembling {self._size} transitions ...")

        idx = 0
        for curr_file in self._episodes:
            curr = load_episode(curr_file)
            length = int(curr_file.stem.split('_')[-3]) #compute_episode_length(curr)

            end = min(idx + length, self._size)

            self._obs[idx:end, :] = curr[DatasetItemType.OBSERVATIONS.value][: end - idx]
            self._actions[idx:end] = curr[DatasetItemType.ACTIONS.value][: end - idx].squeeze(-1)
            self._rewards[idx:end] = curr[DatasetItemType.REWARDS.value][: end - idx].squeeze(-1)
            self._return_to_gos[idx:end] = OfflineDTDataset.compute_return_to_go(curr)[: end - idx].squeeze(-1)
            self._dones[idx:end] = curr[DatasetItemType.DONES.value][: end - idx]
            assert length == (end - idx)
            self._timesteps[idx:end] = np.arange(length)
            idx += length
            if idx >= self._size:
                print(f"[INFO] Finished assembling {self._size} transitions!")
                break

        self._sanity_check()

        self._min_rtgs = self._return_to_gos.min()
        self._max_rtgs = self._return_to_gos.max()
        self.vocab_size = max(self._actions) + 2
        self._padding_action = self.vocab_size - 1
        self._done_idxs = np.argwhere(self._dones).squeeze(-1) + 1

    def _sanity_check(self) -> None:
        """
        Verify the actions in the last episode correspond to the assembled data.
        """
        idx = 0
        for curr in self._episodes:
            length = int(curr.stem.split('_')[-3]) #compute_episode_length(curr)

            if idx + length >= self._size:
                break

            idx += length

        last_episode = load_episode(self._episodes[-1])
        assert np.allclose(
            self._actions[idx:], last_episode[DatasetItemType.ACTIONS.value][: self._size - idx].squeeze(-1)
        ), "[ERROR] Sanity Check fails. Check the assembling logic of transitions."

        print("[DEBUG] Sanity Check succeeded.")

    def get_max_return(self, multiplier: int) -> float:
        return self._max_return * multiplier

    @staticmethod
    def compute_return_to_go(episode: Dict[str, np.ndarray]) -> np.ndarray:
        return np.cumsum(episode[DatasetItemType.REWARDS.value][::-1], axis=0)[::-1]
    
    def _calc_average_return(self) -> float:
        # calculate average return across all episodes
        rewards = []
        for episode in self._episodes:
            # rewards.append(np.sum(episode[DatasetItemType.REWARDS.value]))
            rewards.append(fetch_return_from_path(episode))
        return np.mean(rewards)

    def __len__(self):
        return len(self._obs) - self._block_size

    def __getitem__(self, idx):
        block_size = self._block_size // 3
        done_idx = idx + block_size
        for i, j in enumerate(self._done_idxs):
            if j > idx:  # first done_idx greater than idx
                done_idx = min(int(j), done_idx)
                break

        if i > 0:
            curr_ep_len = self._done_idxs[i] - self._done_idxs[i - 1]
        else:
            curr_ep_len = self._done_idxs[0]

        if curr_ep_len < block_size:
            print("[DEBUG] The trajectory is shorter than the context size. Hence padding this trajectory.....")
            self._short_traj_count += 1
            if i > 0:
                # find episode start index
                idx = self._done_idxs[i - 1]
            else:
                idx = 0
        else:
            idx = done_idx - block_size
        states = torch.tensor(np.array(self._obs[idx:done_idx]), dtype=torch.float32)  # (size, 3, 64, 64)
        # pad states if episode_len  = (done_idx - idx) < block_size
        states = torch.cat(
            [states, torch.zeros(block_size - states.shape[0], *states.shape[1:])], dim=0
        )  # (block_size, 3, 64, 64)
        states = states.reshape(block_size, -1)  # (block_size, 3*64*64)
        states = states / 255.0
        actions = torch.tensor(self._actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        actions = torch.cat(
            [actions, torch.zeros(block_size - actions.shape[0], actions.shape[1]) + self._padding_action], dim=0
        )
        rtgs = self._return_to_gos[idx:done_idx]  # (context size,)
        rtgs = np.pad(rtgs, (0, block_size - rtgs.shape[0]), mode="constant", constant_values=0)
        if self._rtg_noise_prob > 0:
            binary_mask = np.where(np.random.rand(done_idx - idx) < self._rtg_noise_prob)
            # binary_mask[rtgs < 0] = 1
            # rtgs = np.multiply(rtgs, binary_mask)
            random_rtgs = np.random.randint(self._min_rtgs, self._max_rtgs, size=rtgs.shape)
            rtgs[binary_mask] = random_rtgs[binary_mask]

        rtgs = torch.tensor(rtgs, dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self._timesteps[idx : idx + 1], dtype=torch.int64).unsqueeze(1)
        # mask where (done_idx - idx) < block_size is 1 else 0
        padding_mask = torch.cat([torch.ones(done_idx - idx), torch.zeros(block_size - (done_idx - idx))], dim=0)
        return states, actions, rtgs, timesteps, padding_mask




if __name__ == "__main__":
    DATASET_ROOT = "YOUR_DATASET_PATH"
    PROCGEN_ENVS = []
    for env in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, env)):
            PROCGEN_ENVS.append(env)

    print(PROCGEN_ENVS)
    for env in PROCGEN_ENVS:
        print(f"Processing {env}...")
        episodes_dir_path = os.path.join(DATASET_ROOT, env)
        dataset = OfflineDataset(capacity=1000000, episodes_dir_path=episodes_dir_path, percentile=1.0)

        print(env, dataset._calc_average_return())

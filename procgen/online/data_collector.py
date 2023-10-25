# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
from typing import Generator

import numpy as np
import torch
import torch.nn as nn

import baselines
from online.behavior_policies import PPOnet, make_venv
from online.datasets import RolloutStorage, arguments
from utils.utils import LogDirType, LogItemType, set_seed

DEFAULT_CHECKPOINT_DIRECTORY = "YOUR_PPO_CHECKPOINT_DIR"


def collect_data(args):
    print("\nArguments: ", args)
    if args.ratio is not None:
        assert args.ratio >= 0.0 and args.ratio <= 1.0, "The ratio should be between 0 and 1!"

    torch.set_num_threads(1)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Create Envs
    envs = make_venv(
        num_envs=args.num_processes,
        env_name=args.env_name,
        device=device,
        **{
            "num_levels": args.num_levels,
            "start_level": args.start_level,
            "distribution_mode": args.distribution_mode,
            "ret_normalization": False,
            "obs_normalization": False,
        },
    )
    envs.reset()

    # Initialize Model and Replay Buffer
    obs_shape = envs.observation_space.shape
    model = PPOnet(obs_shape, envs.action_space.n, base_kwargs={"hidden_size": args.hidden_size})

    rollouts = RolloutStorage(
        envs.observation_space.shape,
        action_space=envs.action_space,
        n_envs=args.num_processes,
        capacity=args.capacity,
        save_episode=args.save_dataset,
        storage_path=args.dataset_saving_dir,
    )
    if args.ratio is None:
        _roll_out(model, args.seed, envs, rollouts, device, args.checkpoint_path, args.num_env_steps)
    else:
        env_dir = os.path.join(
            DEFAULT_CHECKPOINT_DIRECTORY if args.ratio_checkpoint_dir is None else args.ratio_checkpoint_dir,
            args.env_name,
        )  # env_dir = '/checkpoint/offlinerl/ppo/miner'

        # Sub_env_dirs be like:
        # ['/checkpoint/offlinerl/ppo/miner/xpid_0/', '/checkpoint/offlinerl/ppo/miner/xpid_1/', ...]
        sub_env_dirs = [f.path for f in os.scandir(env_dir) if f.is_dir()]

        for sub_env_dir in sub_env_dirs:
            seed = int(sub_env_dir[-1])
            # Storage path be like: '/checkpoint/offlinerl/ppo/miner/xpid_0/dataset/0.75/'
            rollouts.set_storage_path(os.path.join(sub_env_dir, args.ratio_dataset_dir, str(args.ratio)))
            checkpoint_path = _pick_checkpoint_from_pool(args.ratio, directory=sub_env_dir)
            _roll_out(model, seed, envs, rollouts, device, checkpoint_path, args.num_env_steps)


def _roll_out(
    model: nn.Module,
    seed: int,
    envs: baselines.common.vec_env.VecEnvWrapper,
    rollouts: RolloutStorage,
    device: torch.device,
    checkpoint_path: str,
    target_env_steps: int,
):
    set_seed(seed)

    # Fetch current observations from environment
    obs, _, _, _ = envs.step_wait()
    rollouts.reset()
    rollouts.insert_initial_observations(obs)
    rollouts.to(device)

    # Load checkpoint
    _load_checkpoint(model, checkpoint_path)
    model.to(device)
    print("\n Neural Network: ", model)

    # Roll out the agent and collect Data
    model.eval()
    prev_done_indexes = np.zeros(shape=(envs.num_envs,), dtype=int)
    saved_env_steps = np.zeros(shape=(envs.num_envs,), dtype=int)
    step = 0
    while sum(saved_env_steps) <= target_env_steps:
        # Sample actions
        with torch.no_grad():
            # Raw obs will be saved in rollout storage, while the model needs normalized obs
            # since it was trained with normalized obs.
            _value, action, _action_log_prob = model.act(obs / 255.0)

        # Move one step forward
        obs, rewards, dones, infos = envs.step(action)

        # Store results
        rollouts.insert(obs, action, rewards, dones, infos)

        # Calculate saved env steps
        if any(dones):
            done_env_idxs = np.where(dones == 1)[0]
            saved_env_steps[done_env_idxs] += step - prev_done_indexes[done_env_idxs]
            prev_done_indexes[done_env_idxs] = step

        step += 1


def _load_checkpoint(model: nn.Module, path: str):
    checkpoint_path = os.path.expandvars(os.path.expanduser(f"{path}"))

    print(f"Loading checkpoint from {checkpoint_path} ...")
    try:
        checkpoint_states = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_states[LogItemType.MODEL_STATE_DICT.value])
    except Exception:
        print(f"Unable to load checkpoint from {checkpoint_path}, model is initialized randomly.")


def _pick_checkpoint_from_pool(ratio: float, directory: str) -> pathlib.Path:
    """
    Choose a checkpoint whose expected test return is of ratio (between 0 and 1)
    of that of an expert checkpoint.
    """
    final_checkpoint_path = next(_fetch_final_checkpoint_paths(directory))
    if ratio == 1.0:
        return final_checkpoint_path

    # Fetch all checkpoints with expected test returns
    names = [p for p in _fetch_checkpointed_checkpoint_paths(directory)]
    scores = (_fetch_return_from_name(name) for name in names)
    path_to_score = dict((name, score) for name, score in zip(names, scores))

    # Find the checkpoint closest to the target return.
    target_return = ratio * _fetch_return_from_name(final_checkpoint_path)
    closest_checkpoint = sorted(path_to_score.items(), key=lambda pair: abs(pair[1] - target_return))[0][0]
    return closest_checkpoint


def _fetch_final_checkpoint_paths(directory: str) -> Generator[pathlib.Path, None, None]:
    return _fetch_checkpoint_paths(directory, LogDirType.FINAL)


def _fetch_checkpointed_checkpoint_paths(directory: str) -> Generator[pathlib.Path, None, None]:
    return _fetch_checkpoint_paths(directory, LogDirType.CHECKPOINT)


def _fetch_checkpoint_paths(directory: str, dir_type: LogDirType) -> Generator[pathlib.Path, None, None]:
    return (n for n in pathlib.Path(os.path.join(directory, dir_type.value)).rglob("*.pt"))


def _fetch_return_from_name(name: pathlib.Path) -> float:
    """
    Example:
    >>> name = pathlib.Path('/my_documents/model_1_3.5.pt')
    >>> score = fetch_return_from_name(name)
    >>> score
    3.5
    """
    return float(name.stem.split("_")[-1])


if __name__ == "__main__":
    args = arguments.parser.parse_args()

    collect_data(args)

    print("Data collection completed!")

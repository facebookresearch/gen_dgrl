# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import wandb
from online.behavior_policies import PPOnet, ReplayBuffer, algos, arguments, make_venv
from online.evaluation import evaluate
from utils.utils import LogDirType, LogItemType, set_seed


def _save_model(checkpoint: Dict[str, any], directory: str, filename: str) -> None:
    os.makedirs(directory, exist_ok=True)
    saving_path = os.path.expandvars(os.path.expanduser(f"{directory}/{filename}"))
    torch.save(checkpoint, saving_path)


def _load_model(loading_path: str, model: nn.Module, agent) -> int:
    checkpoint = torch.load(loading_path)

    model.load_state_dict(checkpoint[LogItemType.MODEL_STATE_DICT.value])
    agent.optimizer.load_state_dict(checkpoint[LogItemType.OPTIMIZER_STATE_DICT.value])
    curr_epochs = checkpoint[LogItemType.CURRENT_EPOCH.value]
    return curr_epochs


def train(args):
    print("\nArguments: ", args)

    set_seed(args.seed)
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
            "ret_normalization": True,
            "obs_normalization": True,
        },
    )

    # Initialize Model, Agent, and Replay Buffer
    obs_shape = envs.observation_space.shape
    model = PPOnet(obs_shape, envs.action_space.n, base_kwargs={"hidden_size": args.hidden_size})
    model.to(device)
    print("\n Neural Network: ", model)

    agent = algos.PPO(
        model,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )
    current_epoch = 0
    if args.resume:
        if os.path.exists(args.checkpoint_path):
            print(f"Trying to load checkpoint from {args.checkpoint_path}")
            current_epoch = _load_model(loading_path=args.checkpoint_path, model=model, agent=agent)
        else:
            loading_path = os.path.expandvars(
                os.path.expanduser(
                    os.path.join(args.model_saving_dir, args.env_name, args.xpid, LogDirType.ROLLING.value, "model.pt")
                )
            )
            if os.path.exists(loading_path):
                print(f"Trying to load checkpoint from {loading_path}")
                current_epoch = _load_model(loading_path=loading_path, model=model, agent=agent)
            else:
                print(
                    f"Loading paths do not exist: {args.checkpoint_path}, {loading_path}! \n"
                    + "Will start training from strach!"
                )

        print(f"Resuming checkpoint from Epoch {current_epoch}")

    rollouts = ReplayBuffer(
        obs_shape,
        action_space=envs.action_space,
        n_envs=args.num_processes,
        n_steps=args.num_steps,
        save_episode=args.save_dataset,
        storage_path=os.path.join(args.dataset_saving_dir, args.env_name, "dataset"),
    )
    obs = envs.reset()
    rollouts.insert_initial_observations(obs)
    rollouts.to(device)

    # Training Loop
    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for epoch in range(current_epoch, num_updates):
        model.train()

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = model.act(rollouts._obs_buffer[step])

            obs, reward, done, infos = envs.step(action)

            episode_rewards.extend((info["episode"]["r"] for info in infos if "episode" in info.keys()))

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs, action, reward, done, action_log_prob, value, masks)

        with torch.no_grad():
            next_value = model.get_value(rollouts._obs_buffer[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # Parameters are updated in this step
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if epoch % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (epoch + 1) * args.num_processes * args.num_steps
            print("\n")
            print(f"Update {epoch}, step {total_num_steps}:")
            print(
                f"Last {len(episode_rewards)} training episodes, ",
                f"mean/median reward {np.mean(episode_rewards):.2f}/{np.median(episode_rewards):.2f}",
            )

            eval_episode_rewards = evaluate(args=args, model=model, device=device)
            if args.log_wandb:
                wandb.log(
                    {
                        "step": total_num_steps,
                        "current_update_count": epoch,
                        "policy_gradient_loss": action_loss,
                        "value_loss": value_loss,
                        "dist_entropy": dist_entropy,
                        "Train Mean Episode Returns:": np.mean(episode_rewards),
                        "Train Median Episode Returns:": np.median(episode_rewards),
                        "Test Mean Episode Returns:": np.mean(eval_episode_rewards),
                        "Test Median Episode Returns": np.median(eval_episode_rewards),
                    }
                )

            # save model
            if args.model_saving_dir != "":
                checkpoint = {
                    LogItemType.MODEL_STATE_DICT.value: model.state_dict(),
                    LogItemType.OPTIMIZER_STATE_DICT.value: agent.optimizer.state_dict(),
                    LogItemType.CURRENT_EPOCH.value: epoch + 1,
                }

                _save_model(
                    checkpoint=checkpoint,
                    directory=os.path.join(args.model_saving_dir, args.env_name, args.xpid, LogDirType.ROLLING.value),
                    filename="model.pt",
                )

                if epoch % args.archive_interval == 0:
                    _save_model(
                        checkpoint=checkpoint,
                        directory=os.path.join(
                            args.model_saving_dir, args.env_name, args.xpid, LogDirType.CHECKPOINT.value
                        ),
                        filename=f"model_{epoch}_{np.mean(eval_episode_rewards):.2f}.pt",
                    )

    # Save Final Model
    if args.model_saving_dir != "":
        eval_episode_rewards = evaluate(args=args, model=model, device=device)
        _save_model(
            checkpoint={
                LogItemType.MODEL_STATE_DICT.value: model.state_dict(),
                LogItemType.OPTIMIZER_STATE_DICT.value: agent.optimizer.state_dict(),
            },
            directory=os.path.join(args.model_saving_dir, args.env_name, args.xpid, LogDirType.FINAL.value),
            filename=f"model_{np.mean(eval_episode_rewards):.2f}.pt",
        )


def init_wandb(args):
    if (
        args.wandb_base_url is None
        or args.wandb_api_key is None
        or args.wandb_entity is None
        or args.wandb_project is None
    ):
        arguments.parser.error(
            "Either use '--log_wandb=False' or provide WANDB params ! \n"
            + f"BASE_URL: {args.wandb_base_url}, API_KEY: {args.wandb_api_key}, ENTITY: {args.wandb_entity}"
            + f"PROJECT: {args.wandb_project}"
        )

    os.environ["WANDB_BASE_URL"] = args.wandb_base_url
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args,
        name=args.xpid,
        tags=["vary_n_frames"],
        group=args.xpid[:-2],
    )


if __name__ == "__main__":
    args = arguments.parser.parse_args()

    if args.log_wandb:
        init_wandb(args=args)

    train(args)

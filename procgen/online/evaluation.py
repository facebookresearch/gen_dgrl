import torch
import torch.nn as nn

from online.behavior_policies.envs import make_venv


def evaluate(args, model: nn.Module, device, num_episodes=10):
    model.eval()

    # Sample Levels From the Full Distribution
    eval_envs = make_venv(
        num_envs=1,
        env_name=args.env_name,
        device=device,
        **{
            "num_levels": 200,
            "start_level": 0,
            "distribution_mode": args.distribution_mode,
            "ret_normalization": False,
            "obs_normalization": True,
        },
    )

    eval_episode_rewards = []
    obs = eval_envs.reset()

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _ = model.act(obs)

        obs, _reward, _done, infos = eval_envs.step(action)

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()
    return eval_episode_rewards

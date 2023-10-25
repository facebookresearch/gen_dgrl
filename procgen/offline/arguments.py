import argparse
from utils.utils import str2bool

parser = argparse.ArgumentParser(description="Train offline agents")
parser.add_argument("--algo", type=str, default="bc", choices=["bc", "cql", "dt", "bct", "bcq", "offlinedqn", "iql", "xql"], help="Algorithm to train")
parser.add_argument("--dataset", type=str, default="data/dataset.hdf5", help="Path to dataset")
parser.add_argument("--percentile", type=float, default=1.0, help="percentile for top% training")
parser.add_argument("--dataset_size", type=int, default=1000000, help="Size of dataset")
parser.add_argument("--early_stop", type=str2bool, default=False, help="Use early stopping")

# Model
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--hidden_size", type=int, default=64, help="Size of hidden layers")
parser.add_argument("--agent_model", type=str, default="dqn", choices=["dqn", "bcq", "pporesnetbase", "pporesnet20", "bcqresnetbase"], help="Agent model")
parser.add_argument("--save_path", type=str, default="data/bc.pt", help="Path to save model")
parser.add_argument("--resume", type=str2bool, default=False, help="Resume training")
parser.add_argument("--deterministic", type=str2bool, default=False, help="Sample actions deterministically")
parser.add_argument("--xpid", type=str, default=None, help="experiment name")
parser.add_argument("--eval_eps", type=float, default=0.001, help="epsilon for evaluation")

# Environment
parser.add_argument("--env_name", type=str, default="bigfish", help="Name of environment")
parser.add_argument("--seed", type=int, default=88, help="experiment seed")
parser.add_argument("--num_levels", type=int, default=200, help="number of training levels used in procgen")
parser.add_argument("--distribution_mode", type=str, default="easy", help="Distribution mode of procgen levels")
parser.add_argument("--eval_freq", type=int, default=10, help="frequency for eval")
parser.add_argument("--ckpt_freq", type=int, default=10, help="frequency for checkpointing")

# DDQN
parser.add_argument("--target_update_freq", type=int, default=1000, help="frequency for target network update")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--tau", type=float, default=0.005, help="soft update factor")
parser.add_argument("--buffer_size", type=int, default=1000000, help="size of replay buffer")
parser.add_argument("--eps_start", type=float, default=1.0, help="epsilon start value")
parser.add_argument("--eps_end", type=float, default=0.01, help="epsilon end value")
parser.add_argument("--eps_decay", type=int, default=1000000, help="epsilon decay rate")
parser.add_argument("--perform_polyak_update", type=str2bool, default=False, help="whether to use polyak average or directly copy model weights")

# CQL
parser.add_argument("--cql_alpha", type=float, default=1.0, help="CQL Loss alpha")

# BCQ
parser.add_argument("--bcq_threshold", type=float, default=0.3, help="BCQ threshold for action selection")

# IQL
parser.add_argument("--iql_temperature", type=float, default=0.1, help="IQL temperature for action selection")
parser.add_argument("--iql_expectile", type=float, default=0.8, help="IQL Expectile Loss")

# DT
parser.add_argument("--dt_context_length", type=int, default=128, help="context length for the agent")
parser.add_argument("--grad_norm_clip", type=float, default=0.1, help="gradient norm clip for the agent")
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay only applied on matmul weights")
parser.add_argument("--lr_decay", type=str2bool, default=True, help="learning rate decay with linear warmup followed by cosine decay to 10% of original")
parser.add_argument("--warmup_tokens", type=int, default=10000, help="warmup tokens")
parser.add_argument("--dt_rtg_noise_prob", type=float, default=0.0, help="noise probability for RTGs")
parser.add_argument("--dt_eval_ret", type=int, default=0, help="evaluation return to go. if > 0, then eval rtg = args.dt_eval_ret * max_return in the dataset")

# Single Level Training
parser.add_argument("--capacity_type", type=str, default="transitions", choices=["transitions", "episodes"], help="capacity type")
parser.add_argument("--threshold_metric", type=str, default="median", choices=["percentile", "median"], help="threshold metric")

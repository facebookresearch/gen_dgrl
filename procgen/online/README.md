
This page along with the sub repository `online` provide guidance on how to train an online behavior policy, and on how to use the trained policy to collect data.

> Note 1: Only Procgen is supported.

> Note 2: Only PPO is supported.

## Behavior Policy Training

We use PPO as our behavior policy to collect data for offline RL baseline benchmark.

We have trained and sweeped PPO from scratch on Procgen and stored checkpoints.

If you would like to reproduce the work, please go through the sections below; otherwise if you already have your PPO trained, feel free to 
skip to the next section "Data Collection".

The checkpoints by using the following training script will be stored in directories in ways:
- One directory "rolling": stores 1 checkpoint which is updated during the training. The latest remains. Model name is 'model.pt'.
- One directory "final": stores one single checkpoint at the end of the training, evaluated additionally. Model name is 'model_<AVERAGE_TEST_RETURN_OVER_10_EPISODES>.pt'.
- One directory "checkpoint": stores checkpoints at every logging interval throughout the training process. Model name is 'model_{index}_{AVERAGE_TEST_RETURN_OVER_10_EPISODES>.pt'}


### 1. Ad-hoc training

Specify all the hyper parameters and use the script `trainer` to train the model.

An example command would be like: (better specify the location to save the checkpoints in `model_saving_dir`)

```bash
$ cd .. # to be in the root of this repo. Skip if you already are.
$ python -m online.trainer \
--algo=ppo \
--archive_interval=50 \
--clip_param=0.2 \
--distribution_mode=easy \
--entropy_coef=0.01 \
--env_name=miner \
--eps=1e-05 \
--gae_lambda=0.95 \
--gamma=0.999 \
--hidden_size=256 \
--log_interval=10 \
--lr=0.0005 \
--max_grad_norm=0.5 \
--model_saving_dir=./ppo \
--num_env_steps=5000000 \
--num_levels=200 \
--num_mini_batch=8 \
--num_processes=64 \
--num_steps=256 \
--ppo_epoch=3 \
--resume=True \
--seed=0 \
--start_level=0 \
--value_loss_coef=0.5 \
--xpid=miner-easy-200-ppo-lr0.0005-epoch3-mb8-v0.5-ha0.01_0
```

### 2. Train PPO for all 16 games in Procgen

We have also provided scripts to go over the hyper parameter combinations and do a grid sweep.

All you need to do is to provide a grid config json file under `configs/online/grids/`, say "my_sweep.json", and put list of hyper parameters in it. (check out `configs/online/grids/ppo` for example). You will have to modify `"model_saving_dir"` with an appropriate location on your disk where you'd like to save the checkpoints.

Then, depending on whether you have access to a slurm cluster, you can either

run the following command
```bash
$ cd .. # to be in the root of this repo. Skip if you already are.
# specify grid_config with the name of your grid sweep json file (no extension).
$ python -m train_scripts.cmd_generator --grid_config=my_sweep --num_trials 5 --action SLURM --job_name="my_sweep_of_ppo"
```
if you have access to SLURM Cluster.

OR if you do not have access to SLURM Cluster:

simply genarate the commands and run it in your own way (with `action` as "SAVE", the generated commands will be stored in "my_sweep.txt"):
```bash
$ cd .. # to be in the root of this repo. Skip if you already are.
# specify grid_config with the name of your grid sweep json file (no extension).
$ python -m train_scripts.cmd_generator --grid_config=my_sweep --num_trials 5 --action SAVE 
```

This command will generate all the different combinations (if you have specified more than 1 value for each hyperparameter in the `.json` file) and `--num_trials N` will create `N` number of model seeds for the same set of hyperparameters.

---

## Data Collection

Once we have the trained online behevior policy, we are ready to roll out the agent to collect data!

Dataset is designed as:
- Each completed episode (<state, action, reward, terminate>) is stored in a single .npz file.
- The file name is formated as: `timestamp_<INDEX>_<EPISODE_LENGTH>_<LEVEL_SEED>_<AVERAGE_RETURN_OVER_10_EPISDOES>.npz`. 
  - For example, `20230329T092246_3161_1154_45_40.00.npz` means that 
    - the 3161th saved episode has length of 1154;
    - is generated in a environment with level_seed of 45;
    - the total return of the episode is 40.0.

There are two ways to do that with the script `data_collector`:

### 1. Ad-hoc collection

Specify a checkpoint to start rolling out.

Suggestion:
- Use an integer between 1000 and 5000 for `capacity`, since a trajectory can be as long as 1000, and having a buffer of over 10,000 is not resource friendly.

An example:
```bash
$ cd .. # to be in the root of this repo. Skip if you already are.
$ python -m online.data_collector \
--capacity=2000 \
--checkpoint_path=<YOUR_CHECKPOINT_FILE> \
--dataset_saving_dir=<YOUR_DIR_TO_SAVE_DATASET> \
--env_name=coinrun \
--no_cuda=False \
--num_processes=2 \
--num_env_steps=3000 \
--save_dataset=True \
```

### 2. Collect data for 1 game in Procgen 

This way, the script will go through all the checkpoints for the game that your specify in PPO checkpoints, load those checkpoints and collect data one after another.

The collected data will be saved along with the checkpoints under each xpid for the game, specified by the name `ratio_dataset_dir`.

> An XPID means a string format of hyper parameters, including the seed. It is used in saving checkpoints, and in wandb monitoring.

You will need to provide:
- the directory where your saved PPO checkpoints are (`model_saving_dir` in the previous section);
- a float number between 0 and 1 of the expected test return compared to that of the final checkpoint (expert data):

Suggestion:
- Use an integer between 1000 and 5000 for `capacity`, since a trajectory can be as long as 1000, and having a buffer of over 10,000 is not resource friendly.

An example:
```bash
$ cd .. # to be in the root of this repo. Skip if you already are.
$ python -m online.data_collector \
--capacity=2000 \
--dataset_saving_dir=<YOUR_DIR_TO_SAVE_DATASET> \
--env_name=coinrun \
--no_cuda=False \
--num_processes=64 \
--num_env_steps=1000000 \
--ratio=1.0 \
--ratio_checkpoint_dir=<YOUR_PPO_CHECKPOINTS> \
--ratio_dataset_dir=test_dataset \
--save_dataset=True \
```

## Pretrained Checkpoints

If you would like access to our checkpoints that we had used to collect our datasets for research purposes, kindly reach out to us. We will also be open sourcing these checkpoints soon. 

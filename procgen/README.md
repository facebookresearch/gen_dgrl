# Code for "A Study of Generalization in Offline Reinforcement Learning"- Procgen


## Setup

To install the required packages, run the following commands:

```bash
conda create -n gen-offline python=3.9
conda activate gen-offline

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

To setup procgen - 

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..
```

if there's a numpy issue, try downgrading to version < 1.24: `pip install -U numpy==1.22.4`

---

## Download The Data Set

### ProcGen
To download, use file `download.py`. Using appropriate arguments, you can download the dataset you need. You can comment out the env names if you need to download data for specific games only.

```bash
python download.py --download_folder <path_to_folder> --category_name <type_of_dataset>
```

Optionally, if you want to delete the `*.tar` files after unpacking the dataset, you can pass the following argument: `--clear_archives_after_unpacking`

We provide all datasets used in our paper, collected by pre-trained online policy (PPO):
- [expert_data_1M]: contains 1 million state-action transitions for each of the 16 games. Data is collected by the final PPO checkpoint.
- [expert_data_10M]: contains 10 million state-action transitions for each of the 16 games. Data is collected by the final PPO checkpoint.
- [mixed expert-suboptimal_data_1M]: contains 1 million state-action transitions for each of the 16 games. Data is collected by PPO checkpoints so that the average performance of this dataset is 75% of that of the expert_dataset_1M.
- [suboptimal_data_25M]: contains 25 million state-action transitions for each of the 16 games. Data is collected by using the entire training log of PPO.
- [100k_procgen_dataset_(1 or 40)]: contains 100k state-action transtransitions for each of the 16 games on either level 1 or 40. We provide both, epxert and mixed expert-suboptimal datasets.
  
The decompressed folders contain trajectories grouped by 16 games. Each trajectory is stored as a `.npy` file.

---
# Getting Started
You can find the full list of hyperparameters in the file `offline/arguments.py`

## train_offline_agent.py

E.g. (Replace the `dataset` paths with your proper one! You can optionally pass multiple dataset paths if needed)
```bash
$ python -m offline.train_offline_agent \
--seed=88 \
--dataset=<YOUR_DATASET_PATH> \
--lr=0.0003 \
--batch_size=512 \
--epochs=10 \
--hidden_size=64 \
--agent_model=dqn \
--save_path=<PATH_TO_CHECKPOINT_FOLDER>
```


## Evaluation
To evaluate a previously trained agent, simply copy its training command and replace `offline.train_offline_agent` with `offline.evaluate_offline_agent`.

This code will evaluate the agent on train, validation and test levels and also store the results in a `.csv` file inside the same folder where all the model logs are stored at (i.e. `save_path`)

---

## Running 
To train offline agents on the collected dataset:

- Create an empty `.json` file under `configs/offline/grids/`;
- Put list of hyper parameters in it. (Check the `bc.json` for Behavior Cloning)
- Run the following bash command to generate training commands with all combinations of hyper parameters
For example, 
```bash
$ python -m train_scripts.make_cmd --base_config offline --dir grids --checkpoint --grid_config bc --num_trials 1 --new_line --module_name offline.train_offline_agent 
```
- Run the generated training commands by copying the output

We already provide the training scripts for BC, BCQ, BCT, DT, IQL, CQL in `configs/offline/grids/final` for reproducing our experiments on expert of suboptimal data. To do so, 
1. Add a checkpoint path next to `"save_path"` key
2. Add the path to the dataset next to `"dataset_path"` key


Now generate commands and save them into a file with each argument as a **new line**:
```bash
$ python -m train_scripts.make_cmd --base_config offline --dir grids --checkpoint --grid_config bc --num_trials 3 --new_lin --module_name offline.train_offline_agent >> bc_commands.txt
```

`bc_commands.txt` will contain all the various run commands separated by a new line.

**Optionally**
You can schedule slurm jobs for each command in that file:
```bash
$ python -m train_scripts.slurm -path bc_commands.txt -name "bc" --partition <name of partition> --module_name offline
```

---

The JSON files for training methods using the best hyperparameters settings in each environment for the **1M Expert Dataset** are detailed under `configs/offline/final/`.

## Environments

### [ProcGen](https://github.com/openai/procgen)
![Example mazes](docs/images/procgen_example.png)
| Method   | json config  |
| ------------- |:-------------:|
| BC | `configs/offline/final/bc.json` |
| BCQ | `configs/offline/final/bcq.json` |
| BCT | `configs/offline/final/bct.json` |
| CQL | `configs/offline/final/cql.json` |
| DT | `configs/offline/final/dt.json` |
| IQL | `configs/offline/final/iql.json` |

## Dataset Collection and Online Algorithms
See [online/README.md](online/README.md)

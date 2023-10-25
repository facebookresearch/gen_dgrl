# Offline Learning

This repository contains the source code for the baseline models discussed in the original paper, along with instructions for training the models and running them on WebShop.
## Set Up
* Install additional dependencies via `pip install -r requirements.txt`

## Dataset
* Download the training data for and place it into the `data` folder

We provide labelled human demonstration dataset (collected using `human_trajectories_collector.py`) as well as datasets collected by pre-trained IL policy (collected using `il_data_collector.py`):

[human demonstrations](https://drive.google.com/file/d/1az-NxT6INcPLBxD23K1cDaWr_cWsmVJk/view?usp=sharing): contains 413 episodes from the human demonstration dataset provided by the authors of WebShop, labelled with rewards.

[eval dataset](https://drive.google.com/file/d/1v0gczcw7lc4RAmEd02Gx-rcg1V8w8Shh/view?usp=sharing): contains 54 episodes from the human demonstration dataset which can be used for debugging purposes. Just set `EVAL_PATH` to the path of this file to use this dataset.

[IL dataset of varying size](https://drive.google.com/file/d/1F7HSgeoqlf0VqkMl2ujbGAC1AEh65z8e/view?usp=sharing): contains datasets of various sizes collected by the pretrained IL policy from the original WebShop source code [here](https://drive.google.com/drive/folders/1liZmB1J38yY_zsokJAxRfN8xVO1B_YmD?usp=sharing).

The decompressed dataset consists of a `.jsonl` file which stores all of the episodes.

```bash
cd data
unzip <dataset_path>
cd ..
```
* Download the trained model checkpoints for BART search from [here](https://drive.google.com/drive/folders/1liZmB1J38yY_zsokJAxRfN8xVO1B_YmD?usp=sharing) and store it an appropriate location
```bash
mkdir -p ckpts/web_search/
mv ~/Downloads/search_il_checkpoints_800.zip ckpts/web_search/
unzip ckpts/web_search_il_checkpoints_800.zip
```

## Training
To train, first set the value of `PATH` global variable in each of these files with the path to your `.jsonl` file.

➤ Train the **BC model**
> Open the file below to see the list of arguments.
```bash
python train_choice_il.py
```

➤ Train the **CQL model**
> Open the file below to see the list of arguments.
```bash
python train_choice_cql.py
```

➤ Train the **BCQ model**
> Open the file below to see the list of arguments.
```bash
python train_choice_bcq.py
```

## Reproducing DGRL results
We provide the commands that we ran for reproducing the results in our submission in [here](train_scripts/dgrl_cmds.sh)

## Testing
- Test the model on WebShop:
```bash
python test.py
```
- List of Arguments [here](https://github.com/princeton-nlp/WebShop/blob/master/baseline_models/test.py#L86)
    - `--model_path` should point to the `choice_il_epoch9.pth` file
    - `--bart_path` should point to the `checkpoints-800/` folder
    - `--bcq_alpha` (if evaluating BCQ)

### Notes about Testing
1. You can specify the choice model path (`--model_path`) and the search model path (`--bart_path`) to load different models. 
    
2. While the rule baseline result is deterministic, model results could have variance due to the softmax sampling of the choice policy. `--softmax 0` will use a greedy policy and yield deterministic (but worse) results.

3. `--bart 0` will use the user instruction as the only search query.



import argparse
import json
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
from env import WebEnv
from tqdm import tqdm
from train_rl import parse_args as webenv_args
from transformers import (AutoTokenizer, BartForConditionalGeneration,
                          BartTokenizer)

from baseline_models.models.bert import (BertConfigForWebshop,
                                         BertModelForWebshop)


def bart_predict(input, model, bart_tokenizer, skip_special_tokens=True, **kwargs):
    input_ids = bart_tokenizer(input)['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = model.generate(input_ids, max_length=512, **kwargs)
    return bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=skip_special_tokens)


def process(s):
    s = s.lower().replace('"', '').replace("'", "").strip()
    s = s.replace('[sep]', '[SEP]')
    return s


def process_goal(state):
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    if ', and price lower than' in state:
        state = state.split(', and price lower than')[0]
    return state
    

def data_collator(batch):
    state_input_ids, state_attention_mask, action_input_ids, action_attention_mask, sizes, labels, images = [
    ], [], [], [], [], [], []
    for sample in batch:
        state_input_ids.append(sample['state_input_ids'])
        state_attention_mask.append(sample['state_attention_mask'])
        action_input_ids.extend(sample['action_input_ids'])
        action_attention_mask.extend(sample['action_attention_mask'])
        sizes.append(sample['sizes'])
        labels.append(sample['labels'])
        images.append(sample['images'])
    max_state_len = max(sum(x) for x in state_attention_mask)
    max_action_len = max(sum(x) for x in action_attention_mask)
    return {
        'state_input_ids': torch.tensor(state_input_ids)[:, :max_state_len],
        'state_attention_mask': torch.tensor(state_attention_mask)[:, :max_state_len],
        'action_input_ids': torch.tensor(action_input_ids)[:, :max_action_len],
        'action_attention_mask': torch.tensor(action_attention_mask)[:, :max_action_len],
        'sizes': torch.tensor(sizes),
        'images': torch.tensor(images),
        'labels': torch.tensor(labels),
    }


def predict(obs, info, model, tokenizer, softmax=False, rule=False, bart_model=None):
    valid_acts = info['valid']
    if valid_acts[0].startswith('search['):
        if bart_model is None:
            return valid_acts[-1]
        else:
            goal = process_goal(obs)
            query = bart_predict(goal, bart_model, bart_tokenizer=bart_tokenizer, num_return_sequences=5, num_beams=5)
            # query = random.choice(query)  # in the paper, we sample from the top-5 generated results.
            query = query[0]  #... but use the top-1 generated search will lead to better results than the paper results.
            return f'search[{query}]'
            
    if rule:
        item_acts = [act for act in valid_acts if act.startswith('click[item - ')]
        if item_acts:
            return item_acts[0]
        else:
            assert 'click[buy now]' in valid_acts
            return 'click[buy now]'
                
    state_encodings = tokenizer(process(obs), max_length=512, truncation=True, padding='max_length')
    action_encodings = tokenizer(list(map(process, valid_acts)), max_length=512, truncation=True,  padding='max_length')
    if 'image_feat' in info:
        images = info['image_feat'].tolist()
    else:
        images = [[0.] * 512] * len(obs)
    batch = {
        'state_input_ids': state_encodings['input_ids'],
        'state_attention_mask': state_encodings['attention_mask'],
        'action_input_ids': action_encodings['input_ids'],
        'action_attention_mask': action_encodings['attention_mask'],
        'sizes': len(valid_acts),
        'images': images,
        'labels': 0
    }
    batch = data_collator([batch])
    # make batch cuda
    batch = {k: v.cuda() for k, v in batch.items()}
    outputs = model(**batch)
    if softmax:
        idx = torch.multinomial(F.softmax(outputs.logits[0], dim=0), 1)[0].item()
    else:
        idx = outputs.logits[0].argmax(0).item()
    return valid_acts[idx]


# Setup
tokenizer = AutoTokenizer.from_pretrained(
    'bert-base-uncased', truncation_side='left')
print(len(tokenizer))
tokenizer.add_tokens(['[button]', '[button_]', '[clicked button]',
                     '[clicked button_]'], special_tokens=True)
print(len(tokenizer))


bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
bart_path = "./ckpts/web_search/checkpoint-800/"
bart_model = BartForConditionalGeneration.from_pretrained(bart_path)
print('bart model loaded', bart_path)


model_path = "./ckpts/web_click/epoch_9/model.pth"
config = BertConfigForWebshop(image=False)
model = BertModelForWebshop(config)
model.cuda()
model.load_state_dict(torch.load(model_path), strict=False)
print('bert il model loaded', model_path)


# Env
env_args = webenv_args()[0]

random.seed(233)

train_env = WebEnv(env_args, split='train', id='train_')
server = train_env.env.server
eval_env = WebEnv(env_args, split='eval', id='eval_', server=server)
test_env = WebEnv(env_args, split='test', id='test_', server=server)
print('env loaded')

train_env.env.num_prev_obs = 0
train_env.env.num_prev_actions = 0
eval_env.env.num_prev_obs = 0
eval_env.env.num_prev_actions = 0
test_env.env.num_prev_obs = 0
test_env.env.num_prev_actions = 0
print('no memory')

envs = {"train": train_env, "eval": eval_env, "test": test_env}


parser = argparse.ArgumentParser()
parser.add_argument('--num_traj', default=1, type=int)
args, _ = parser.parse_known_args()
print(f"# Trajectories: {args.num_traj}")


# Rollout
def episode(model, env, level=None, verbose=False, softmax=False, rule=False, bart_model=None):
    trajectory = defaultdict(list)
    obs, info = env.reset(level)
    trajectory["states"].append(obs)
    trajectory["available_actions"].append([])

    def assemble_action_index(trajectory):
        # Assemble action index
        for index in range(len(trajectory["actions"])):
            try:
                action_idx = trajectory["available_actions"][index].index(trajectory["actions"][index])
            except Exception:
                # First valid action space is empty, so the index will always be -1.
                action_idx = -1
            trajectory["action_idxs"].append(action_idx)
        return trajectory

    def remove_last_state_and_valid_action_space(trajectory):
        trajectory["states"] = trajectory["states"][:-1]
        trajectory["available_actions"] = trajectory["available_actions"][:-1]
        return trajectory

    def post_rollout(trajectory):
        trajectory = assemble_action_index(trajectory)
        trajectory = remove_last_state_and_valid_action_space(trajectory)
        return trajectory

    for i in range(100):
        action = predict(obs, info, model, tokenizer=tokenizer, softmax=softmax, rule=rule, bart_model=bart_model)
        if verbose:
            print(action)

        obs, reward, done, info = env.step(action)

        trajectory["actions"].append(action)
        trajectory["states"].append(obs)
        trajectory["rewards"].append(reward)
        trajectory["available_actions"].append(info["valid"])

        if done:
            trajectory = post_rollout(trajectory)
            return trajectory, reward

    trajectory = post_rollout(trajectory)
    return trajectory, 0


num_traj = args.num_traj
episodes = {name: [] for name, env in envs.items()}
for name, env in envs.items():
    print(f"Env: {name} - Collecting {num_traj} trajectories.")
    for i in tqdm(range(num_traj)):
        traj, _ = episode(model, env=env, softmax=True, bart_model=bart_model)
        episodes[name].append(traj)

# Dump
to_dump = episodes["train"] + episodes["eval"] + episodes["test"]
print(len(to_dump))

PATH = f'./il_trajectories_{num_traj}.jsonl'
with open(PATH, 'w') as outfile:
    for entry in to_dump:
        json.dump(entry, outfile)
        outfile.write('\n')

with open(PATH, 'r') as json_file:
    json_list = list(json_file)

print(f"Length - {len(json_list)}")

sample = random.sample(range(num_traj), k=1)[0]
print(json.loads(json_list[sample])["states"][0])

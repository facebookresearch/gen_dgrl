import numpy as np
import json
import argparse
from env import WebEnv

def parse_args():
    parser = argparse.ArgumentParser()
    # logging
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output_dir', default='logs')
    parser.add_argument('--ckpt_freq', default=10000, type=int)
    parser.add_argument('--eval_freq', default=500, type=int)
    parser.add_argument('--test_freq', default=5000, type=int)
    parser.add_argument('--log_freq', default=100, type=int)
    parser.add_argument('--wandb', default=1, type=int)

    # rl
    parser.add_argument('--num_envs', default=4, type=int)
    parser.add_argument('--step_limit', default=100, type=int)
    parser.add_argument('--max_steps', default=300000, type=int)
    parser.add_argument('--learning_rate', default=1e-5, type=float)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--clip', default=10, type=float)
    parser.add_argument('--bptt', default=8, type=int)
    parser.add_argument('--exploration_method', default='softmax', type=str, choices=['eps', 'softmax'])
    parser.add_argument('--w_pg', default=1, type=float)
    parser.add_argument('--w_td', default=1, type=float)
    parser.add_argument('--w_il', default=0, type=float)
    parser.add_argument('--w_en', default=1, type=float)

    # model
    parser.add_argument('--network', default='bert', type=str, choices=['bert', 'rnn'])
    parser.add_argument('--bert_path', default="", type=str, help='which bert to load')
    parser.add_argument('--embedding_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--grad_encoder', default=1, type=int)
    parser.add_argument('--get_image', default=0, type=int, help='use image in models')

    # env
    parser.add_argument('--num', default=None, type=int)
    parser.add_argument('--click_item_name', default=1, type=int)
    parser.add_argument('--state_format', default='text_rich', type=str)
    parser.add_argument('--human_goals', default=1, type=int, help='use human goals')
    parser.add_argument('--num_prev_obs', default=0, type=int, help='number of previous observations')
    parser.add_argument('--num_prev_actions', default=0, type=int, help='number of previous actions')
    parser.add_argument('--extra_search_path', default="./data/goal_query_predict.json", type=str, help='path for extra search queries')
    

    # experimental 
    parser.add_argument('--ban_buy', default=0, type=int, help='ban buy action before selecting options')
    parser.add_argument('--score_handicap', default=0, type=int, help='provide score in state')
    parser.add_argument('--go_to_item', default=0, type=int)
    parser.add_argument('--go_to_search', default=0, type=int)
    parser.add_argument('--harsh_reward', default=0, type=int)


    parser.add_argument('--debug', default=0, type=int, help='debug mode')
    parser.add_argument("--f", help="a dummy argument to fool ipython", default="1")

    return parser.parse_known_args()


def strip_website(state):
    """Remove environment names."""
    state = state.lower()#.replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    return state

def new_process_goal(state):
    """Copied from train_choice_ii.py, but would not strip the price part in the instruction."""
    state = state.lower().replace('"', '').replace("'", "")
    state = state.replace('amazon shopping game\ninstruction:', '').replace('webshop\ninstruction:', '')
    state = state.replace('\n[button] search [button_]', '').strip()
    # if ', and price lower than' in state:
    #     state = state.split(', and price lower than')[0]
    return state

def dice(env):
    """Pick an initial state which is present in the dataset."""
    while True:
        ob, info = env.reset()
        try:
            index = json_goals.index(new_process_goal(ob))
            return index, ob, info
        except:
            pass

def check_states_in_episode(env, ob, info, data) -> bool:
    """Compare the states from the rollout with those in the dataset"""
    all_obs = [ob.encode('utf-8')]
    all_rewards = []
    all_dones = []
    all_infos = [info]

    for action in data["actions"]:
        new_ob, reward, done, new_info = env.step(action)

        all_obs.append(new_ob.encode('utf-8'))
        all_rewards.append(reward)
        all_dones.append(done)
        all_infos.append(new_info)

    
    for i in range(1, len(data["states"])):
        assert strip_website(all_obs[i].decode('utf-8')) == strip_website(data["states"][i])

    return all_rewards
        

# Setup
args, _ = parse_args()
env = WebEnv(args, split='train', id='train_')

PATH = "./data/il_trajs_finalized_images.jsonl"
with open(PATH, 'r') as json_file:
    json_list = list(json_file)

json_goals = [new_process_goal(json.loads(json_str)["states"][0]) for json_str in json_list]

assert len(json_goals) == len(json_list)

# Roll out
count = 0
chosen_episodes = np.zeros((len(json_goals),))
episode_with_rewards = [None] * len(json_goals)

while True:
    index, ob, info = dice(env)
    if chosen_episodes[index]:
        continue

    print(index)

    chosen_episodes[index] = 1
    episode_from_dataset = json.loads(json_list[index])
    rewards = check_states_in_episode(env, ob=ob, info=info, data=episode_from_dataset)
    episode_with_rewards[index] = rewards

    count += 1
    if count >= 412: # It takes 20mins to collect 413 trajectories with reward.
        break

# Dump
output_path = './human_trajectories.jsonl'
output = []
for json_str, rewards in zip(json_list, episode_with_rewards):
    if rewards is not None:
        episode = json.loads(json_str)
        episode["rewards"] = rewards
        output.append(episode)
    
with open(output_path, 'w') as outfile:
    for entry in output:
        json.dump(entry, outfile)
        outfile.write('\n')




# load WebEnV
import csv
import os
import sys
import json
from train_rl import parse_args as webenv_args
from env import WebEnv  # TODO: just use webshopEnv?


# load Model
from train_choice_il import *
from models.cql import QModelForWebshop, CQLConfigForWebshop
from models.bcq import BCQConfigForWebshop, BCQModelForWebshop
from transformers import BartForConditionalGeneration, BartTokenizer
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')


import random

def bart_predict(input, model, skip_special_tokens=True, **kwargs):
    input_ids = bart_tokenizer(input)['input_ids']
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output = model.generate(input_ids, max_length=512, **kwargs)
    return bart_tokenizer.batch_decode(output.tolist(), skip_special_tokens=skip_special_tokens)


def predict(obs, info, model, algo, softmax=False, rule=False, bart_model=None, bcq_alpha=None):
    valid_acts = info['valid']
    if valid_acts[0].startswith('search['):
        if bart_model is None:
            return valid_acts[-1]
        else:
            goal = process_goal(obs)
            query = bart_predict(goal, bart_model, num_return_sequences=5, num_beams=5)
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
    batch = {
        'state_input_ids': state_encodings['input_ids'],
        'state_attention_mask': state_encodings['attention_mask'],
        'action_input_ids': action_encodings['input_ids'],
        'action_attention_mask': action_encodings['attention_mask'],
        'sizes': len(valid_acts),
        'images': info['image_feat'].tolist(),
        'labels': 0
    }
    batch = data_collator([batch])
    # make batch cuda
    batch = {k: v.cuda() for k, v in batch.items()}
    if algo == 'il':
        outputs = model(**batch)
        if softmax:
            idx = torch.multinomial(F.softmax(outputs.logits[0], dim=0), 1)[0].item()
        else:
            idx = outputs.logits[0].argmax(0).item()
    elif algo == 'cql':
        output_q_vals = model(batch["state_input_ids"], batch["state_attention_mask"], batch["action_input_ids"], batch["action_attention_mask"], batch["sizes"], images=batch["images"], labels=None)
        idx = torch.stack([logit.argmax(dim=0)
                                      for logit in output_q_vals])
    elif algo == 'bcq':
        assert bcq_alpha is not None
        eval_q_vals, eval_action_probs, _ = model(batch["state_input_ids"], batch["state_attention_mask"], batch["action_input_ids"], batch["action_attention_mask"], batch["sizes"], images=batch["images"], labels=None)
        eval_action_probs_exp = [probs.exp() for probs in eval_action_probs]
        eval_action_probs_norm = [(probs/probs.max() > bcq_alpha).float() for probs in eval_action_probs_exp]
        # Use large negative number to mask actions from argmax
        idx = torch.stack([(act_prob * q_val + (1. - act_prob) * -1e8).argmax()
                                for q_val, act_prob in zip(eval_q_vals, eval_action_probs_norm)])
    else:
        return NotImplementedError
    return valid_acts[idx]



def episode(env, model, algo, idx=None, verbose=False, softmax=False, rule=False, bart_model=None, bcq_alpha=None):
    obs, info = env.reset(idx)
    if verbose:
        print(info['goal'])
    for i in range(100):
        action = predict(obs, info, model, algo, softmax=softmax, rule=rule, bart_model=bart_model, bcq_alpha=bcq_alpha)
        if verbose:
            print(action)
        obs, reward, done, info = env.step(action)
        if done:
            return reward
    return 0



def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--model_path", type=str, default="./ckpts/web_click/epoch_9/model.pth", help="Where to store the final model.")
    parser.add_argument("--mem", type=int, default=0, help="State with memory")
    parser.add_argument("--bart_path", type=str, default='./ckpts/web_search/checkpoint-800', help="BART model path if using it")
    parser.add_argument("--bart", type=bool, default=True, help="Flag to specify whether to use bart or not (default: True)")
    parser.add_argument("--image", type=bool, default=False, help="Flag to specify whether to use image or not (default: True)")
    parser.add_argument("--softmax", type=bool, default=True, help="Flag to specify whether to use softmax sampling or not (default: True)")
    parser.add_argument("--rule", type=bool, default=False, help="Flag to specify whether to use rule or not (default: False)")
    parser.add_argument("--algo", type=str, default="il", choices=['il','cql','bcq'], help="Type of algorthm to run")
    parser.add_argument("--bcq_alpha", type=float, default=0.5, help="Alpha value for BCQ")
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    env_args = webenv_args()[0]
    test_env = WebEnv(env_args, split='test')
    eval_env = WebEnv(env_args, split='eval')
    train_env = WebEnv(env_args, split='train')
    print('env loaded')
    
    if args.mem:
        test_env.env.num_prev_obs = 1
        test_env.env.num_prev_actions = 5
        
        eval_env.env.num_prev_obs = 1
        eval_env.env.num_prev_actions = 5
        
        train_env.env.num_prev_obs = 1
        train_env.env.num_prev_actions = 5
    
        print('memory')
    else:
        test_env.env.num_prev_obs = 0
        test_env.env.num_prev_actions = 0
        
        eval_env.env.num_prev_obs = 0
        eval_env.env.num_prev_actions = 0
        
        train_env.env.num_prev_obs = 0
        train_env.env.num_prev_actions = 0
        
        print('no memory')
    

    if args.bart:
        bart_model = BartForConditionalGeneration.from_pretrained(args.bart_path)
        print('bart model loaded', args.bart_path)
    else:
        bart_model = None


    if args.algo == 'il':
        config = BertConfigForWebshop(image=args.image)
        model = BertModelForWebshop(config)
    elif args.algo == 'cql':
        config = CQLConfigForWebshop(image=args.image)
        model = QModelForWebshop(config)
    elif args.algo == 'bcq':
        config = BCQConfigForWebshop(image=args.image)
        model = BCQModelForWebshop(config)
    model.cuda()
    model.load_state_dict(torch.load(args.model_path), strict=False)
    print(f'bert {args.algo} model loaded', args.model_path)
    
    print("test RETURN")
    print('idx | reward (model), reward (rule)')
    scores_softmax, scores_rule = [], []
    for i in range(500):
        score_softmax = episode(test_env, model, args.algo, idx=i, softmax=args.softmax, bart_model=bart_model, bcq_alpha=args.bcq_alpha)
        score_rule = episode(test_env, model, args.algo, idx=i, rule=True, bcq_alpha=args.bcq_alpha)
        print(i, '|', score_softmax * 10, score_rule * 10)  # env score is 0-10, paper is 0-100
        scores_softmax.append(score_softmax)
        scores_rule.append(score_rule)
    score_softmax = sum(scores_softmax) / len(scores_softmax)
    score_rule = sum(scores_rule) / len(scores_rule)
    harsh_softmax = len([s for s in scores_softmax if s == 10.0])
    harsh_rule = len([s for s in scores_rule if s == 10.0])
    print('------')
    print('avg test score (model, rule):', score_softmax * 10, score_rule * 10)
    print('avg test success rate % (model, rule):', harsh_softmax / 500 * 100, harsh_rule / 500 * 100)
    test_scores = [score_softmax*10, score_rule*10, harsh_softmax / 500 * 100, harsh_rule / 500 * 100]
    
    root_dir = os.path.dirname(args.model_path)
    with open(root_dir + '/test_scores.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['score', 'rule', 'harsh', 'rule_harsh'])
        writer.writerow(test_scores)
        
    print("eval RETURN")
    print('idx | reward (model), reward (rule)')
    scores_softmax, scores_rule = [], []
    for i in range(500, 1500):
        score_softmax, score_rule = episode(eval_env, model, algo=args.algo, idx=i, softmax=args.softmax, bart_model=bart_model, bcq_alpha=args.bcq_alpha), episode(eval_env, model, algo=args.algo, idx=i, rule=True, bcq_alpha=args.bcq_alpha)
        print(i, '|', score_softmax * 10, score_rule * 10)  # env score is 0-10, paper is 0-100
        scores_softmax.append(score_softmax)
        scores_rule.append(score_rule)
    score_softmax = sum(scores_softmax) / len(scores_softmax)
    score_rule = sum(scores_rule) / len(scores_rule)
    harsh_softmax = len([s for s in scores_softmax if s == 10.0])
    harsh_rule = len([s for s in scores_rule if s == 10.0])
    print('------')
    print('avg eval score (model, rule):', score_softmax * 10, score_rule * 10)
    print('avg eval success rate % (model, rule):', harsh_softmax / 500 * 100, harsh_rule / 500 * 100)
    eval_scores = [score_softmax*10, score_rule*10, harsh_softmax / 500 * 100, harsh_rule / 500 * 100]
    
    root_dir = os.path.dirname(args.model_path)
    with open(root_dir + '/eval_scores.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['score', 'rule', 'harsh', 'rule_harsh'])
        writer.writerow(eval_scores)
        
    print("TRAIN RETURN")
    print('idx | reward (model), reward (rule)')
    scores_softmax, scores_rule = [], []
    for i in range(1500, 2000):
        score_softmax, score_rule = episode(train_env, model, algo=args.algo, idx=i, softmax=args.softmax, bart_model=bart_model, bcq_alpha=args.bcq_alpha), episode(train_env, model, algo=args.algo, idx=i, rule=True, bcq_alpha=args.bcq_alpha)
        print(i, '|', score_softmax * 10, score_rule * 10)  # env score is 0-10, paper is 0-100
        scores_softmax.append(score_softmax)
        scores_rule.append(score_rule)
    score_softmax = sum(scores_softmax) / len(scores_softmax)
    score_rule = sum(scores_rule) / len(scores_rule)
    harsh_softmax = len([s for s in scores_softmax if s == 10.0])
    harsh_rule = len([s for s in scores_rule if s == 10.0])
    print('------')
    print('avg train score (model, rule):', score_softmax * 10, score_rule * 10)
    print('avg train success rate % (model, rule):', harsh_softmax / 500 * 100, harsh_rule / 500 * 100)
    train_scores = [score_softmax*10, score_rule*10, harsh_softmax / 500 * 100, harsh_rule / 500 * 100]
    
    root_dir = os.path.dirname(args.model_path)
    with open(root_dir + '/train_scores.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['score', 'rule', 'harsh', 'rule_harsh'])
        writer.writerow(train_scores)
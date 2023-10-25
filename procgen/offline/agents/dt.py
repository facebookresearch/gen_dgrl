"""
This file contains training loop implementing Decision Transformer.

Source:
1. https://github.com/kzl/decision-transformer/blob/master/atari/mingpt
2. https://github.com/karpathy/minGPT
3. https://github.com/karpathy/nanoGPT

----------------------------------------------------------------------------------------------------------------------------

MIT License

Copyright (c) 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors (https://arxiv.org/abs/2106.01345)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import torch
import torch.nn as nn
import torch.nn.functional as  F
from torch.distributions import Categorical
from utils.gpt_arch import GPT, GPTConfig
import numpy as np
import math


class DTConfig:
    # optimization parameters
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
class DecisionTransformer:
    def __init__(self, 
                 observation_space, 
                 action_space,  
                 agent_model, 
                 train_data_vocab_size,
                 train_data_block_size,
                 max_timesteps,
                 context_len,
                 dataset_size,
                 lr=6e-4,
                 betas=(0.9, 0.95),
                 grad_norm_clip=1.0,
                 weight_decay=0.1,
                 lr_decay=True,
                 warmup_tokens=512*20):
        """
        Initialize the agent.

        :param observation_space: the observation space for the environment
        :param action_space: the action space for the environment
        :param lr: the learning rate for the agent
        :param agent_model: the agent model type: [reward_conditioned (DT), naive (BCT)]
        :param train_data_vocab_size: the size of the vocabulary for the agent
        :param train_data_block_size: the block size for the agent
        :param max_timesteps: the number of timesteps for the agent
        :param context_len: the context length for the agent
        :param dataset_size: the dataset size for the agent
        :param betas: the betas for the agent
        :param grad_norm_clip: the gradient norm clip for the agent
        :param weight_decay: the weight decay only applied on matmul weights
        :param lr_decay: the learning rate decay with linear warmup followed by cosine decay to 10% of original
        :param warmup_tokens: the warmup tokens
        :param final_tokens: the final tokens (at what point we reach 10% of original LR)
        """
        
        # Initialise GPT config and GPT model
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.agent_model = agent_model
        
        self.train_data_vocab_size = train_data_vocab_size
        self.train_data_block_size = train_data_block_size
        self.model_type = "reward_conditioned" if agent_model == "dt_reward_conditioned" else "naive"
        self.max_timesteps = max_timesteps
        self.final_tokens=2*dataset_size*context_len*3
        
        self.betas = betas
        self.grad_norm_clip = grad_norm_clip
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens
        
        self.mconf = GPTConfig(self.train_data_vocab_size, self.train_data_block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=self.model_type, max_timestep=self.max_timesteps, inp_channels=3)
        self.model = GPT(self.mconf)
        
        self.config = DTConfig(learning_rate=self.lr, lr_decay=self.lr_decay, warmup_tokens=self.warmup_tokens, final_tokens=self.final_tokens,
                      num_workers=4, model_type=self.model_type, max_timestep=self.max_timesteps)
        
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = self.raw_model.configure_optimizers(self.config)
        
        self.total_steps = 0
        self.tokens = 0
        
    def set_device(self, device):
        self.model.to(device)
        
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        
    def top_k_logits(self, logits, k):
        # Source: https://github.com/kzl/decision-transformer/blob/master/atari/mingpt/utils.py
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out
    
    def entropy(self, logits, probs):
        '''
        Calculates mean entropy
        Source: https://pytorch.org/docs/stable/_modules/torch/distributions/categorical.html#Categorical.entropy
        '''
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        p_log_p = logits * probs
        return -p_log_p.sum(-1)

    def get_categorical_entropy(self, prob):
        # print(prob.shape)
        distb = Categorical(torch.tensor(prob))
        return distb.entropy()

    @torch.no_grad()
    def sample(self, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None, return_probs=False):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        
        Source: https://github.com/kzl/decision-transformer/blob/master/atari/mingpt/utils.py
        """
        block_size = self.model.get_block_size()
        self.model.eval()
        for k in range(steps):
            # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
            x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
            if actions is not None:
                actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
            rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
            logits, _ = self.model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            # x = torch.cat((x, ix), dim=1)
            x = ix

        if return_probs:
            bc_entropy = self.get_categorical_entropy(probs)
            return x, probs, bc_entropy
        return x
    
    def train_step(self, observations, actions, rtgs, timesteps, padding_mask):

        _, loss = self.model(observations, actions, actions, rtgs, timesteps, padding_mask)
        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
        
        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        
        # decay the learning rate based on our progress
        if self.config.lr_decay:
            self.tokens += (actions >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
            if self.tokens < self.config.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self.tokens - self.config.warmup_tokens) / float(max(1, self.config.final_tokens - self.config.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.config.learning_rate * lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.config.learning_rate
        
        self.total_steps += 1
        
        # create stats dict
        stats = {"loss": loss.item(), "lr": lr, "total_steps": self.total_steps}
        return stats
    
    def save(self, num_epochs, path):
        """
        Save the model to a given path.

        :param path: the path to save the model
        """
        save_dict = {
            "model_save_dict": self.model.state_dict(),
            "tokens_processed": self.tokens,
            "total_steps": self.total_steps,
            "curr_epochs": num_epochs
        }
        torch.save(save_dict, path)
        return
    
    def load(self, path):
        """
        Load the model from a given path.

        :param path: the path to load the model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_save_dict"])
        self.tokens = checkpoint["tokens_processed"]
        self.total_steps = checkpoint["total_steps"]
        
        return checkpoint["curr_epochs"]
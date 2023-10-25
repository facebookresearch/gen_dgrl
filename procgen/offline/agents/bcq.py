# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as  F
import numpy as np
import math

from utils import AGENT_CLASSES

class BCQ:
	def __init__(self, 
				 observation_space, 
				 action_space, 
				 lr, 
				 agent_model, 
				 hidden_size, 
				 gamma, 
				 target_update_freq, 
				 tau,
				 eps_start, 
				 eps_end, 
				 eps_decay, 
				 bcq_threshold,
				 perform_polyak_update):
		"""
		Initialize the agent.

		:param observation_space: the observation space for the environment
		:param action_space: the action space for the environment
		:param lr: the learning rate for the agent
		:param hidden_size: the size of the hidden layers for the agent
		:param gamma: the discount factor for the agent
		:param target_update_freq: the frequency with which to update the target network
		:param tau: the soft update factor for the target network
		:param eps_start: the starting epsilon for the agent
		:param eps_end: the ending epsilon for the agent
		:param eps_decay: the decay rate for epsilon
		:param bcq_threshold: the threshold for selecting the best action
		:param perform_polyak_update: whether to use polyak averaging or not
		"""
		self.observation_space = observation_space
		self.action_space = action_space
		self.lr = lr
		self.hidden_size = hidden_size
		self.gamma = gamma
		self.target_update_freq = target_update_freq
		self.tau = tau
		
		self.model = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.target_model = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		
		self.target_model.load_state_dict(self.model.state_dict())
		self.target_model.eval()
		
		self.total_steps = 0
		
		self.eps_start = eps_start
		self.eps_end = eps_end
		self.eps_decay = eps_decay
		self.bcq_threshold = bcq_threshold
		self.perform_polyak_update = perform_polyak_update
		
	def train(self):
		self.model.train()

	def eval(self):
		self.model.eval()
		
	def set_device(self, device):
		self.model.to(device)
		self.target_model.to(device)
		
	def train_step(self, observations, actions, rewards, next_observations, dones):
		with torch.no_grad():
			# Q-values for best actions in next observations
			next_q_values_model, next_action_probs, _ = self.model(next_observations) # [batch_size, num_actions]
			next_action_probs = next_action_probs.exp()
			next_action_probs = (next_action_probs/next_action_probs.max(1, keepdim=True)[0] > self.bcq_threshold).float()
			# Use large negative number to mask actions from argmax
			next_actions = (next_action_probs * next_q_values_model + (1 - next_action_probs) * -1e8).argmax(1, keepdim=True) # [batch_size, 1]
			next_q_values, _, _ = self.target_model(next_observations)
			next_q_value = next_q_values.gather(1, next_actions).reshape(-1, 1)
			# Compute the target of the current Q-values
			target_q_values = rewards + (1 - dones) * self.gamma * next_q_value # [batch_size, 1]
			
		# Q-values for current observations
		q_values, curr_action_probs, curr_action_i = self.model(observations)
		# Compute the predicted q values for the actions taken
		pred_q_values = q_values.gather(1, actions)
		
		# Train the model with Bellman error as targets
		ddqn_loss = F.smooth_l1_loss(pred_q_values, target_q_values)
		
		# Calculate BCQ loss
		i_loss = F.nll_loss(curr_action_probs, actions.reshape(-1))
		bcq_loss = i_loss + 1e-2 * curr_action_i.pow(2).mean()
		
		loss = ddqn_loss + bcq_loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		# Update the target network, copying all weights and biases in DQN
		if self.perform_polyak_update:
			self.soft_update_target(tau=self.tau)
		else:
			if self.total_steps % self.target_update_freq == 0:
				self.copy_update_target()
		
		self.total_steps += 1
		
		# create stats dict
		stats = {"loss": loss.item(), "ddqn_loss": ddqn_loss.item(), "bcq_loss": bcq_loss.item(), "total_steps": self.total_steps}
		return stats
	
	def soft_update_target(self, tau):
		"""
		Perform a soft update of the target network.
		:param tau: the soft update coefficient
		"""
		for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
			
	def copy_update_target(self):
		"""
		Perform a duplicate update of the target network.
		"""
		self.target_model.load_state_dict(self.model.state_dict())
			
	@property
	def calculate_eps(self):
		"""
		Calculate epsilon given the current timestep, initial epsilon, end epsilon and decay rate, where initial_eps > end_eps.
		"""
		
		eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.total_steps / self.eps_decay)
		return eps
	
	def eval_step(self, observations, eps=0.001):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		self.model.eval()
		# Epsilon-greedy action selection
		if np.random.uniform(0,1) <= eps:
			action = np.random.randint(self.action_space, size=(1,))
		else:
			with torch.no_grad():
				q_values, action_probs, _ = self.model(observations)
				action_probs = action_probs.exp()
				action_probs = (action_probs/action_probs.max(1, keepdim=True)[0] > self.bcq_threshold).float()
				# Use large negative number to mask actions from argmax
				action = (action_probs * q_values + (1. - action_probs) * -1e8).argmax(1).detach().cpu().numpy()
		return action
	
	def save(self, num_epochs, path):
		"""
		Save the model to a given path.

		:param path: the path to save the model
		"""
		save_dict = {
			"model_state_dict": self.model.state_dict(),
			"target_model_state_dict": self.target_model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
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
		self.model.load_state_dict(checkpoint["model_state_dict"])
		self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
		self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		self.total_steps = checkpoint["total_steps"]
		self.target_model.eval()
		return checkpoint["curr_epochs"]
			

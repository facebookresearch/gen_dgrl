# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from online.behavior_policies.distributions import Categorical
from utils import AGENT_CLASSES


class IQL:
	def __init__(
		self,
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
		iql_temperature,
		iql_expectile,
		perform_polyak_update
	):
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
		:param iql_temperature: the temperature for the IQL agent
		:param iql_expectile: the expectile weight for the IQL agent
		"""

		# Implement Implicit Q Learning, which has an Actor, Critic and Q Function
		self.observation_space = observation_space
		self.action_space = action_space
		self.lr = lr
		self.hidden_size = hidden_size
		self.gamma = gamma
		self.target_update_freq = target_update_freq
		self.tau = tau

		self.total_steps = 0

		self.eps_start = eps_start
		self.eps_end = eps_end
		self.eps_decay = eps_decay

		self.iql_temperature = iql_temperature
		self.iql_expectile = iql_expectile
		
		self.perform_polyak_update = perform_polyak_update

		# Implement the Actor, Critic and Value Function
		self.model_actor = AGENT_CLASSES[agent_model](
			observation_space, action_space, hidden_size, use_actor_linear=False
		)
		self.actor_dist = Categorical(hidden_size, self.action_space)
		# optimizer_actor uses parameters from model_actor and actor_dist
		actor_model_params = list(self.model_actor.parameters()) + list(self.actor_dist.parameters())
		self.optimizer_actor = torch.optim.Adam(actor_model_params, lr=self.lr)

		self.model_v = AGENT_CLASSES[agent_model](observation_space, 1, hidden_size)
		self.optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr=self.lr)

		self.model_q1 = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.optimizer_q1 = torch.optim.Adam(self.model_q1.parameters(), lr=self.lr)
		self.target_q1 = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.target_q1.load_state_dict(self.model_q1.state_dict())
		self.target_q1.eval()

		self.model_q2 = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.optimizer_q2 = torch.optim.Adam(self.model_q2.parameters(), lr=self.lr)
		self.target_q2 = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.target_q2.load_state_dict(self.model_q2.state_dict())
		self.target_q2.eval()

	def train(self):
		self.model_actor.train()
		self.actor_dist.train()
		self.model_v.train()
		self.model_q1.train()
		self.model_q2.train()

	def eval(self):
		self.model_actor.eval()
		self.actor_dist.eval()
		self.model_v.eval()
		self.model_q1.eval()
		self.model_q2.eval()

	def set_device(self, device):
		self.model_actor.to(device)
		self.actor_dist.to(device)
		self.model_v.to(device)
		self.model_q1.to(device)
		self.model_q2.to(device)
		self.target_q1.to(device)
		self.target_q2.to(device)

	def expectile_loss(self, u_diff, expectile=0.8):
		"""
		Calculate the expectile loss for the IQL agent.

		:param value: the value function shape [batch_size, 1]
		:param Q_value: the Q-value shape [batch_size, 1]
		:param expectile: the expectile weight
		"""
		# expectile_weight = torch.where(u_diff > 0, expectile, 1 - expectile)  # [batch_size, 1]
		# L2_tau = expectile_weight * (u_diff**2)  # [batch_size, 1]
		return torch.mean(torch.abs(expectile - (u_diff < 0).float()) * u_diff**2)

	def train_step(self, observations, actions, rewards, next_observations, dones):
		# 1. Calculate Value Loss
		with torch.no_grad():
			q1 = self.target_q1(observations).gather(1, actions)  # [batch_size, 1]
			q2 = self.target_q2(observations).gather(1, actions)  # [batch_size, 1]
			q_minimum = torch.min(q1, q2)  # [batch_size, 1]

		curr_value = self.model_v(observations)  # [batch_size, 1]
		u_diff = q_minimum - curr_value  # [batch_size, 1]
		value_loss = self.expectile_loss(u_diff, self.iql_expectile)  # [1]
		self.optimizer_v.zero_grad(set_to_none=True)
		value_loss.backward()
		self.optimizer_v.step()

		# 2. Calculate Critic Loss
		with torch.no_grad():
			next_v = self.model_v(next_observations)  # [batch_size, 1]
		target_q = rewards + (1 - dones) * self.gamma * next_v.detach()  # [batch_size, 1]
		curr_q1 = self.model_q1(observations).gather(1, actions)  # [batch_size, 1]
		curr_q2 = self.model_q2(observations).gather(1, actions)  # [batch_size, 1]
		critic1_loss = F.mse_loss(curr_q1, target_q).mean()  # [1]

		self.optimizer_q1.zero_grad(set_to_none=True)
		critic1_loss.backward()
		self.optimizer_q1.step()

		critic2_loss = F.mse_loss(curr_q2, target_q).mean()  # [1]
		self.optimizer_q2.zero_grad(set_to_none=True)
		critic2_loss.backward()
		self.optimizer_q2.step()
		
		# Update the target network, copying all weights and biases in DQN
		if self.perform_polyak_update:
			for target_param, param in zip(self.target_q1.parameters(), self.model_q1.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for target_param, param in zip(self.target_q2.parameters(), self.model_q2.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else:
			if self.total_steps % self.target_update_freq == 0:
				self.target_q1.load_state_dict(self.model_q1.state_dict())
				self.target_q2.load_state_dict(self.model_q2.state_dict())

		# 3. Calculate Actor Loss
		exp_action = torch.exp(u_diff.detach() * self.iql_temperature)  # [batch_size, 1]
		# take minimum of exp_action and 100.0 to avoid overflow
		exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
		# _, action_log_prob = self.get_action(observations, return_log_probs=True)  # [batch_size, 1]
		action_feats = self.model_actor(observations)  # [batch_size, 512]
		action_dist = self.actor_dist(action_feats)
		action_log_prob = action_dist.log_probs(actions)
		actor_loss = -(exp_action * action_log_prob).mean()  # [1]
		self.optimizer_actor.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.optimizer_actor.step()

		self.total_steps += 1

		# create stats dict
		with torch.no_grad():
			loss = value_loss + actor_loss + critic1_loss + critic2_loss
		stats = {
			"loss": loss.item(),
			"value_loss": value_loss.item(),
			"critic1_loss": critic1_loss.item(),
			"critic2_loss": critic2_loss.item(),
			"actor_loss": actor_loss.item(),
			"total_steps": self.total_steps,
		}
		# print(stats["actor_loss"])
		return stats

	@property
	def calculate_eps(self):
		"""
		Calculate epsilon given the current timestep, initial epsilon, end epsilon and decay rate.
		"""
		eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.total_steps / self.eps_decay)
		return eps

	def eval_step(self, observations, eps=0.0, return_log_probs=False):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		deterministic = eps == 0.0
		with torch.no_grad():
			action_feats = self.model_actor(observations)
			action_dist = self.actor_dist(action_feats)

			if deterministic:
				action = action_dist.mode()
			else:
				action = action_dist.sample()  # [batch_size, 1]

			action_log_prob = action_dist.log_probs(action)

		if return_log_probs:
			return action.cpu().numpy(), action_log_prob

		return action.cpu().numpy()

	def get_action(self, observations, eps=0.0, return_log_probs=False):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		deterministic = eps == 0.0

		action_feats = self.model_actor(observations)  # [batch_size, 512]
		action_dist = self.actor_dist(action_feats)

		if deterministic:
			action = action_dist.mode()
		else:
			action = action_dist.sample()  # [batch_size, 1]

		action_log_prob = action_dist.log_probs(action)
		# st()
		# print(action_log_prob)

		if return_log_probs:
			return action, action_log_prob

		return action

	def save(self, num_epochs, path):
		"""
		Save the model to a given path.

		:param path: the path to save the model
		"""
		save_dict = {
			"actor_state_dict": self.model_actor.state_dict(),
			"actor_dist_state_dict": self.actor_dist.state_dict(),
			"model_v_state_dict": self.model_v.state_dict(),
			"model_q1_state_dict": self.model_q1.state_dict(),
			"model_q2_state_dict": self.model_q2.state_dict(),
			"target_q1_state_dict": self.target_q1.state_dict(),
			"target_q2_state_dict": self.target_q2.state_dict(),
			"optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
			"optimizer_v_state_dict": self.optimizer_v.state_dict(),
			"optimizer_q1_state_dict": self.optimizer_q1.state_dict(),
			"optimizer_q2_state_dict": self.optimizer_q2.state_dict(),
			"total_steps": self.total_steps,
			"curr_epochs": num_epochs,
		}
		torch.save(save_dict, path)
		return

	def load(self, path):
		"""
		Load the model from a given path.

		:param path: the path to load the model
		"""
		checkpoint = torch.load(path)
		self.model_actor.load_state_dict(checkpoint["actor_state_dict"])
		self.actor_dist.load_state_dict(checkpoint["actor_dist_state_dict"])
		self.model_v.load_state_dict(checkpoint["model_v_state_dict"])
		self.model_q1.load_state_dict(checkpoint["model_q1_state_dict"])
		self.model_q2.load_state_dict(checkpoint["model_q2_state_dict"])
		self.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
		self.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])
		self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
		self.optimizer_v.load_state_dict(checkpoint["optimizer_v_state_dict"])
		self.optimizer_q1.load_state_dict(checkpoint["optimizer_q1_state_dict"])
		self.optimizer_q2.load_state_dict(checkpoint["optimizer_q2_state_dict"])
		self.total_steps = checkpoint["total_steps"]
		self.target_q1.eval()
		self.target_q2.eval()

		return checkpoint["curr_epochs"]

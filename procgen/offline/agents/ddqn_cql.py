import torch
import torch.nn as nn
import torch.nn.functional as  F
import numpy as np
import math

from utils import AGENT_CLASSES

class CQL:
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
				 cql_alpha,
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
		:param cq_alpha: the alpha value for CQL
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
		self.cql_alpha = cql_alpha
		self.perform_polyak_update = perform_polyak_update
		
	def train(self):
		self.model.train()

	def eval(self):
		self.model.eval()
		
	def set_device(self, device):
		self.model.to(device)
		self.target_model.to(device)
		
	def train_step(self, observations, actions, rewards, next_observations, dones):
		# Q-values for current observations
		q_values = self.model(observations) # [batch_size, num_actions]
		with torch.no_grad():
			# Q-values for best actions in next observations
			next_q_values = self.target_model(next_observations)
			next_actions = torch.argmax(self.model(next_observations), dim=1).unsqueeze(1) # [batch_size, 1]
			next_q_value = next_q_values.gather(1, next_actions) # [batch_size, 1]
			# Compute the target of the current Q-values
			target_q_values = rewards + (1 - dones) * self.gamma * next_q_value
		# Compute the predicted q values for the actions taken
		pred_q_values = q_values.gather(1, actions)
		
		# Train the model with Bellman error as targets
		ddqn_loss = F.smooth_l1_loss(pred_q_values, target_q_values)
		
		# Calculate CQL loss
		logsumexp_q_values = torch.logsumexp(q_values, dim=1, keepdim=True) # [batch_size, 1]
		one_hot_actions = F.one_hot(actions.squeeze(dim=1), self.action_space) # [batch_size, num_actions]
		q_values_selected = torch.sum(q_values * one_hot_actions, dim=1, keepdim=True) # [batch_size, 1]
		cql_loss = self.cql_alpha * torch.mean(logsumexp_q_values - q_values_selected)
		
		loss = ddqn_loss + cql_loss
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
		stats = {"loss": loss.item(), "ddqn_loss": ddqn_loss.item(), "cql_loss": cql_loss.item(), "eps": self.calculate_eps, "total_steps": self.total_steps}
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
		Calculate epsilon given the current timestep, initial epsilon, end epsilon and decay rate.
		"""
		eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.total_steps / self.eps_decay)
		return eps
	
	def eval_step(self, observations, eps=0.5):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		self.model.eval()
		# Epsilon-greedy action selection
		if np.random.uniform(0,1) < eps:
			action = np.random.randint(self.action_space, size=(1,))
		else:
			with torch.no_grad():
				q_values = self.model(observations)
				action = torch.argmax(q_values, dim=1).detach().cpu().numpy()
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
			
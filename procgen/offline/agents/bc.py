import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import AGENT_CLASSES
from online.behavior_policies.distributions import Categorical


class BehavioralCloning:
    def __init__(self, observation_space, action_space, lr, agent_model, hidden_size=64):
        """
        Initialize the agent.

        :param observation_space: the observation space for the environment
        :param action_space: the action space for the environment
        :param lr: the learning rate for the agent
        :param hidden_size: the size of the hidden layers for the agent
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.hidden_size = hidden_size

        self.model_base = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size, use_actor_linear=False)
        self.model_dist = Categorical(hidden_size, self.action_space)
        self.optimizer = torch.optim.Adam(list(self.model_base.parameters()) + list(self.model_dist.parameters()), lr=self.lr)
        
        self.total_steps = 0

    def train(self):
        self.model_base.train()
        self.model_dist.train()

    def eval(self):
        self.model_base.eval()
        self.model_dist.eval()

    def set_device(self, device):
        self.model_base.to(device)
        self.model_dist.to(device)

    def eval_step(self, observation, eps=0.0):
        """
        Given an observation, return an action.

        :param observation: the observation for the environment
        :return: the action for the environment in numpy
        """
        deterministic = eps == 0.0
        with torch.no_grad():
            actor_features = self.model_base(observation)
            dist = self.model_dist(actor_features)
            
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        return action.cpu().numpy()

    def train_step(self, observations, actions, rewards, next_observations, dones):
        """
        Update the agent given observations and actions.

        :param observations: the observations for the environment
        :param actions: the actions for the environment
        """
        # squeeze actions to [batch_size] if they are [batch_size, 1]
        if len(actions.shape) == 2:
            actions = actions.squeeze(dim=1)
            
        actor_features = self.model_base(observations)
        dist = self.model_dist(actor_features)
        action_log_probs = dist._get_log_softmax()
        
        self.optimizer.zero_grad()
        loss = F.nll_loss(action_log_probs, actions)
        loss.backward()
        self.optimizer.step()
        self.total_steps += 1
        # create stats dict
        stats = {"loss": loss.item(), "total_steps": self.total_steps}
        return stats

    def save(self, num_epochs, path):
        """
        Save the model to a given path.

        :param path: the path to save the model
        """
        save_dict = {
            "model_base_state_dict": self.model_base.state_dict(),
            "model_dist_state_dict": self.model_dist.state_dict(),
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
        self.model_base.load_state_dict(checkpoint["model_base_state_dict"])
        self.model_dist.load_state_dict(checkpoint["model_dist_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        return checkpoint["curr_epochs"]

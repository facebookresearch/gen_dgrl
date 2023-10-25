"""
This file contains implementation of various encoding architectures made of MLPs, CNNs, Transformers, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import init

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))

def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DQNEncoder(nn.Module):
    # Implements DQN 3-layer CNN encoder
    def __init__(self, observation_space, action_space=15, hidden_size=64, use_actor_linear=True):
        super(DQNEncoder, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(observation_space.shape[-1], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, hidden_size, 4, stride=2)
        self.conv3 = nn.Conv2d(hidden_size, 64, 3, stride=1)
        self.fc = nn.Linear(1024, action_space)
        
        apply_init_(self.modules())

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class IQLCritic(nn.Module):
    # implements IQL critic network
    def __init__(self, base_class, observation_space, action_space=15, hidden_size=64):
        super(IQLCritic, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.critic_feats = 256

        self.critic_linear = nn.Linear(action_space + self.critic_feats, 1)
        self.base = PPOResNetBaseEncoder(observation_space, self.critic_feats, hidden_size)
        
        apply_init_(self.modules())

    def forward(self, x, a):
        """_summary_

        Args:
            x: observation of shape (batch_size, *observation_space.shape)
            a: actions of shape (batch_size, 1)

        Returns:
            value of shape (batch_size, 1)
        """
        x = self.base(x)  # (batch_size, critic_feats)
        q_val = self.critic_linear(torch.cat([x, a], dim=1))  # (batch_size, 1)
        return q_val


class BCQEncoder(nn.Module):
    # Source: https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/discrete_BCQ.py#LL8-L31C48
    def __init__(self, observation_space, action_space=15, hidden_size=64):
        super(BCQEncoder, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(observation_space.shape[-1], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, hidden_size, 4, stride=2)
        self.conv3 = nn.Conv2d(hidden_size, 64, 3, stride=1)

        self.q1 = nn.Linear(1024, 512)
        self.q2 = nn.Linear(512, action_space)

        self.i1 = nn.Linear(1024, 512)
        self.i2 = nn.Linear(512, action_space)
        
        apply_init_(self.modules())

    def forward(self, x):
        c = F.relu(self.conv1(x))
        c = F.relu(self.conv2(c))
        c = F.relu(self.conv3(c))

        q = F.relu(self.q1(c.reshape(-1, 1024)))
        i = F.relu(self.i1(c.reshape(-1, 1024)))
        i = self.i2(i)
        return self.q2(q), F.log_softmax(i, dim=1), i


class Flatten(nn.Module):
    """
    Flatten a tensor
    """

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """

    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """

    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class PPOResNetBaseEncoder(NNBase):
    """
    Residual Network from PPO implementation -> 1M parameters
    """

    def __init__(self, observation_space, action_space=15, hidden_size=256, channels=[16, 32, 32], use_actor_linear=True):
        super(PPOResNetBaseEncoder, self).__init__(hidden_size)
        self.observation_space = observation_space
        self.use_actor_linear = use_actor_linear

        self.layer1 = self._make_layer(observation_space.shape[-1], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        if self.use_actor_linear:
            self.actor_linear = init_(nn.Linear(hidden_size, action_space))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.use_actor_linear:
            return self.actor_linear(x)
        
        return x


class PPOResNet20Encoder(NNBase):
    """
    Residual Network with 20 layers -> 35M parameters
    """

    def __init__(self, observation_space, action_space=15, hidden_size=256, channels=[64, 256, 256, 512], use_actor_linear=True):
        super(PPOResNet20Encoder, self).__init__(hidden_size)
        self.observation_space = observation_space
        self.use_actor_linear = use_actor_linear
        
        self.layer1 = self._make_layer(observation_space.shape[-1], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.layer4 = self._make_layer(channels[2], channels[3])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc1 = init_relu_(nn.Linear(8192, 2048))
        self.fc2 = init_relu_(nn.Linear(2048, hidden_size))

        if self.use_actor_linear:
            self.actor_linear = init_(nn.Linear(hidden_size, action_space))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        if self.use_actor_linear:
            return self.actor_linear(x)
        
        return x

class BCQResnetBaseEncoder(NNBase):
    """
    BCQ Netwrok with Residual Network-style encoder -> 1M parameters
    """
    def __init__(self, observation_space, action_space=15, hidden_size=64, channels=[16, 32, 32], use_actor_linear=False):
        super(BCQResnetBaseEncoder, self).__init__(hidden_size)
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.conv1 = self._make_layer(observation_space.shape[-1], channels[0])
        self.conv2 = self._make_layer(channels[0], channels[1])
        self.conv3 = self._make_layer(channels[1], channels[2])

        self.q1 = nn.Linear(2048, hidden_size)
        self.q2 = nn.Linear(hidden_size, action_space)

        self.i1 = nn.Linear(2048, hidden_size)
        self.i2 = nn.Linear(hidden_size, action_space)
        
        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        c = F.relu(self.conv1(x))
        c = F.relu(self.conv2(c))
        c = F.relu(self.conv3(c))

        q = F.relu(self.q1(c.reshape(-1, 2048)))
        i = F.relu(self.i1(c.reshape(-1, 2048)))
        i = self.i2(i)
        return self.q2(q), F.log_softmax(i, dim=1), i
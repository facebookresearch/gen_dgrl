# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .archs import DQNEncoder, BCQEncoder, PPOResNetBaseEncoder, PPOResNet20Encoder, BCQResnetBaseEncoder

AGENT_CLASSES = {
    "dqn": DQNEncoder,
    "bcq": BCQEncoder,
    "pporesnetbase": PPOResNetBaseEncoder,
    "pporesnet20": PPOResNet20Encoder,
    "bcqresnetbase": BCQResnetBaseEncoder,
}

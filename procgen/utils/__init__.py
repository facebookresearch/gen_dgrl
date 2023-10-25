from .archs import DQNEncoder, BCQEncoder, PPOResNetBaseEncoder, PPOResNet20Encoder, BCQResnetBaseEncoder

AGENT_CLASSES = {
    "dqn": DQNEncoder,
    "bcq": BCQEncoder,
    "pporesnetbase": PPOResNetBaseEncoder,
    "pporesnet20": PPOResNet20Encoder,
    "bcqresnetbase": BCQResnetBaseEncoder,
}

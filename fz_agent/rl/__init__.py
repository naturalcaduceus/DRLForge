"""fz_agent.rl —— 强化学习算法包。

统一导出 build_rl_algorithm 和 BaseRLAlgo。
"""
from .base import BaseRLAlgo
from .ppo import PPO, ActorCritic
from .dqn import DQN, QNetwork, ReplayBuffer
from .ddqn import DDQN
from .dueling_dqn import DuelingDQN, DuelingQNetwork
from .rainbow import Rainbow
from .ddpg import DDPG
from .sac import SAC
from .td3 import TD3
from .a2c import A2C
from .reinforce import REINFORCE
from .iqn import IQN

__all__ = [
    "BaseRLAlgo", "build_rl_algorithm",
    "PPO", "ActorCritic",
    "DQN", "QNetwork", "ReplayBuffer",
    "DDQN",
    "DuelingDQN", "DuelingQNetwork",
    "Rainbow", "DDPG", "SAC", "TD3", "A2C", "REINFORCE", "IQN",
]


ALGO_REGISTRY = {
    "ppo": PPO,
    "dqn": DQN,
    "ddqn": DDQN,
    "dueling_dqn": DuelingDQN,
    "rainbow": Rainbow,
    "ddpg": DDPG,
    "sac": SAC,
    "td3": TD3,
    "a2c": A2C,
    "reinforce": REINFORCE,
    "iqn": IQN,
}


def build_rl_algorithm(algo_name: str, cfg: dict, *args, **kwargs):
    """按名字构建 RL 算法实例。

    Args:
        algo_name: 如 "ppo", "dqn"（大小写不敏感）
        cfg:       算法超参 dict
        *args, **kwargs: 透传给算法 __init__（一般是 policy_net 等）
    """
    key = algo_name.lower()
    if key not in ALGO_REGISTRY:
        raise ValueError(
            f"未知算法 '{algo_name}'，可用: {list(ALGO_REGISTRY.keys())}"
        )
    cls = ALGO_REGISTRY[key]
    return cls(cfg, *args, **kwargs)

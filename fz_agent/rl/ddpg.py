"""DDPG：深度确定性策略梯度（骨架）。"""
from .base import BaseRLAlgo


class DDPG(BaseRLAlgo):
    def __init__(self, config, actor, critic, target_actor=None, target_critic=None):
        super().__init__(config, actor, target_actor)
        self.critic = critic
        self.target_critic = target_critic

    def select_action(self, obs):
        raise NotImplementedError("DDPG 骨架待实现")

    def update(self, batch):
        raise NotImplementedError("DDPG 骨架待实现")

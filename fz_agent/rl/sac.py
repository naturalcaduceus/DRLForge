"""SAC：Soft Actor-Critic（骨架）。"""
from .base import BaseRLAlgo


class SAC(BaseRLAlgo):
    def __init__(self, config, actor, q1, q2, target_q1, target_q2):
        super().__init__(config, actor)
        self.q1 = q1; self.q2 = q2
        self.target_q1 = target_q1; self.target_q2 = target_q2

    def select_action(self, obs):
        raise NotImplementedError("SAC 骨架待实现")

    def update(self, batch):
        raise NotImplementedError("SAC 骨架待实现")

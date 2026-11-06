"""A2C：Advantage Actor-Critic（骨架）。"""
from .base import BaseRLAlgo


class A2C(BaseRLAlgo):
    def select_action(self, obs):
        raise NotImplementedError("A2C 骨架待实现")

    def update(self, batch):
        raise NotImplementedError("A2C 骨架待实现")

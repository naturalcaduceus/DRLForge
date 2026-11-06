"""IQN：Implicit Quantile Network（骨架）。"""
from .base import BaseRLAlgo


class IQN(BaseRLAlgo):
    def select_action(self, obs):
        raise NotImplementedError("IQN 骨架待实现")

    def update(self, batch):
        raise NotImplementedError("IQN 骨架待实现")

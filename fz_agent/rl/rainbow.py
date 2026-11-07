"""Rainbow：6 种 DQN 改进的合体（骨架，待完整实现）。"""
from .base import BaseRLAlgo


class Rainbow(BaseRLAlgo):
    def select_action(self, obs):
        raise NotImplementedError("Rainbow 骨架待实现")

    def update(self, batch):
        raise NotImplementedError("Rainbow 骨架待实现")

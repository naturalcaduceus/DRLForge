"""REINFORCE：蒙特卡洛策略梯度（骨架）。"""
from .base import BaseRLAlgo


class REINFORCE(BaseRLAlgo):
    def select_action(self, obs):
        raise NotImplementedError("REINFORCE 骨架待实现")

    def update(self, batch):
        raise NotImplementedError("REINFORCE 骨架待实现")

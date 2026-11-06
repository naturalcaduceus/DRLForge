"""TD3：Twin Delayed DDPG（骨架）。"""
from .base import BaseRLAlgo


class TD3(BaseRLAlgo):
    def __init__(self, config, actor, q1, q2, target_actor, target_q1, target_q2):
        super().__init__(config, actor, target_actor)
        self.q1 = q1; self.q2 = q2
        self.target_q1 = target_q1; self.target_q2 = target_q2

    def select_action(self, obs):
        raise NotImplementedError("TD3 骨架待实现")

    def update(self, batch):
        raise NotImplementedError("TD3 骨架待实现")

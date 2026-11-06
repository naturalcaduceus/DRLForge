"""DDQN：Double DQN，使用 policy net 选择动作，target net 评估 Q 值。"""
import torch.nn as nn
from .dqn import DQN, QNetwork
from .base import BaseRLAlgo


class DDQN(DQN):
    """继承 DQN，只重写 update 中的 target Q 计算逻辑。"""

    def update(self, batch):
        import torch
        from .dqn import ReplayBuffer

        if not isinstance(batch, ReplayBuffer):
            raise TypeError("DDQN.update 期望输入 ReplayBuffer 实例")

        batch_size = self.cfg.get("batch_size", 64)
        if len(batch) < batch_size:
            return None

        obs, actions, rewards, next_obs, dones = batch.sample(batch_size)
        import numpy as np
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        acts_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1)
        rew_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy(obs_t).gather(1, acts_t).squeeze(-1)

        # Double: policy 选动作，target 给 Q 值
        with torch.no_grad():
            next_actions = self.policy(next_t).argmax(dim=-1, keepdim=True)
            next_q = self.target(next_t).gather(1, next_actions).squeeze(-1)
            target_q = rew_t + self.gamma * next_q * (1.0 - done_t)

        import torch.nn.functional as F
        loss = F.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        import torch.nn as nn
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self._soft_update()
        self._decay_epsilon()

        return {"q_loss": loss.item(), "mean_q": q_values.mean().item()}

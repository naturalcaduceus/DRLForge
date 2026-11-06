"""DQN 算法：Q-Network + Experience Replay + 软更新目标网络。"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from .base import BaseRLAlgo


class QNetwork(nn.Module):
    """Q 网络，针对实体集合观测做了 masked mean pooling。"""

    def __init__(self, obs_dim=3, hidden_dim=128, action_dim=3):
        super().__init__()

        self.entity_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        valid_mask = torch.any(torch.abs(obs) > 1e-8, dim=-1)
        entity_features = self.entity_encoder(obs)

        mask = valid_mask.unsqueeze(-1).to(entity_features.dtype)
        feature_sum = (entity_features * mask).sum(dim=1)
        valid_count = mask.sum(dim=1).clamp(min=1.0)
        pooled = feature_sum / valid_count

        shared_feature = self.shared(pooled)
        return self.q_head(shared_feature)


class ReplayBuffer:
    """简单的均匀经验回放池。"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = map(np.stack, zip(*batch))
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)


class DQN(BaseRLAlgo):
    """DQN + epsilon-greedy + 软更新目标网络。"""

    def __init__(self, config: dict, policy_net: nn.Module,
                 target_net: nn.Module = None):
        import copy
        # 如果没有显式 target_net，就用 deepcopy 克隆一个（通用、不管网络具体类型）
        if target_net is None:
            target_net = copy.deepcopy(policy_net)
            for p in target_net.parameters():
                p.requires_grad = False

        super().__init__(config, policy_net, target_net)

        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.max_grad_norm = config.get("max_grad_norm", 10.0)

        self.epsilon = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.05)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        lr = config.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    # ------- 对外接口 -------

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """选择动作。单样本返回 int，批量返回 tensor。"""
        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device,
        )
        batched = obs_tensor.dim() != 2
        if not batched:
            obs_tensor = obs_tensor.unsqueeze(0)

        q_values = self.policy(obs_tensor)
        actions = torch.argmax(q_values, dim=-1)

        if not deterministic and not batched and random.random() < self.epsilon:
            action_dim = q_values.shape[-1]
            return random.randint(0, action_dim - 1)

        if not batched:
            return int(actions.item())
        return actions

    # ------- 基类接口 -------

    def select_action(self, obs: torch.Tensor):
        return self.act(obs, deterministic=False)

    def update(self, batch):
        """从 replay buffer 采样并更新。batch 应为 ReplayBuffer 实例。"""
        if not isinstance(batch, ReplayBuffer):
            raise TypeError("DQN.update 期望输入 ReplayBuffer 实例")

        batch_size = self.cfg.get("batch_size", 64)
        if len(batch) < batch_size:
            return None

        obs, actions, rewards, next_obs, dones = batch.sample(batch_size)

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            actions, dtype=torch.long, device=self.device,
        ).unsqueeze(-1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(
            next_obs, dtype=torch.float32, device=self.device,
        )
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Q(s, a)
        q_values = self.policy(obs).gather(1, actions).squeeze(-1)

        # r + gamma * max Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target(next_obs).max(dim=-1)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self._soft_update()
        self._decay_epsilon()

        return {"q_loss": loss.item(), "mean_q": q_values.mean().item()}

    def _soft_update(self):
        for p, tp in zip(self.policy.parameters(), self.target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

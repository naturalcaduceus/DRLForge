"""Rainbow：6 种 DQN 改进的合体。
Rainbow DQN 是 6 种 DQN 改进的合体，
这 6 种改进分别是：分布强化学习（Categorical/Distributional RL）、
N 步回报（N-step Returns）、
优先经验回放、Noisy Net、
双 Q 学习
以及竞争网络。
为了完整实现 Rainbow 算法，我们需要在骨架中引入这些机制。
以下是一个基于 PyTorch 的 Rainbow 算法完整实现框架，
包含了核心的数学逻辑和网络结构：
"""
from .base import BaseRLAlgo

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 假设 V_min, V_max, N_ATOMS, N_STEP, GAMMA 等超参数已在外部定义
# V_min, V_max: 分布强化学习中回报的上下界
# N_ATOMS: 离散分布的原子数
# N_STEP: 多步回报的步数
# GAMMA: 折扣因子

class NoisyLinear(nn.Module):
    """Noisy Net 层：将确定性参数和随机噪声参数结合，实现探索"""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # 测试阶段不使用噪声
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowNetwork(nn.Module):
    """竞争网络 + Noisy Net + 分布式输出"""
    def __init__(self, obs_dim, action_dim, n_atoms, v_min, v_max):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.register_buffer('supports', torch.linspace(v_min, v_max, n_atoms))
        
        # 竞争网络 分为 Value 和 Advantage 两流
        self.feature = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU())
        
        self.value_stream = nn.Sequential(
            NoisyLinear(128, 128), nn.ReLU(),
            NoisyLinear(128, n_atoms)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(128, 128), nn.ReLU(),
            NoisyLinear(128, action_dim * n_atoms)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features).unsqueeze(1)  # (batch, 1, n_atoms)
        advantages = self.advantage_stream(features).view(-1, self.action_dim, self.n_atoms)  # (batch, action_dim, n_atoms)
        
        # 竞争网络聚合，并使用 softmax 输出概率分布
        q_dist = values + advantages - advantages.mean(1, keepdim=True)
        prob_dist = F.softmax(q_dist, dim=-1)
        # 防止概率为0导致log(0)
        prob_dist = prob_dist.clamp(min=1e-5)
        return prob_dist
    
    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class Rainbow(BaseRLAlgo):
    def __init__(self, obs_dim, action_dim, v_min, v_max, n_atoms, n_step, gamma, lr):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.n_step = n_step
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        
        self.online_net = RainbowNetwork(obs_dim, action_dim, n_atoms, v_min, v_max)
        self.target_net = RainbowNetwork(obs_dim, action_dim, n_atoms, v_min, v_max)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

    def select_action(self, obs):
        """选择动作：基于分布式输出的期望 Q 值"""
        # 将观测放入网络
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        self.online_net.eval()
        with torch.no_grad():
            prob_dist = self.online_net(obs)  # (1, action_dim, n_atoms)
            # 计算每个动作的期望 Q 值：Q = sum(p_i * z_i)
            q_values = (prob_dist * self.online_net.supports).sum(-1)  # (1, action_dim)
            action = q_values.argmax(1).item()
        self.online_net.train()
        return action

    def update(self, batch):
        """更新网络：结合 N步回报、双Q学习、分布式投影"""
        obs, actions, rewards, next_obs, dones = batch
        
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 1. 计算当前状态的动作概率分布
        prob_dist = self.online_net(obs)  # (batch, action_dim, n_atoms)
        # 选取对应动作的分布
        current_probs = prob_dist[torch.arange(len(actions)), actions]  # (batch, n_atoms)
        
        # 2. 双 Q 学习：用 online_net 选动作，用 target_net 评估
        with torch.no_grad():
            next_prob_dist_online = self.online_net(next_obs)
            next_q_values = (next_prob_dist_online * self.online_net.supports).sum(-1)
            next_actions = next_q_values.argmax(1)  # (batch,)
            
            next_prob_dist_target = self.target_net(next_obs)
            next_probs = next_prob_dist_target[torch.arange(len(next_actions)), next_actions]  # (batch, n_atoms)

        # 3. 分布式投影
        # 计算 N 步回报的目标分布支撑点: Tz = R + (gamma^n) * Z
        # 假设 batch 中的 reward 已经是 n_step 累积好的
        supports = self.target_net.supports.unsqueeze(0)  # (1, n_atoms)
        Tz = rewards.unsqueeze(1) + (self.gamma ** self.n_step) * supports * (1 - dones.unsqueeze(1))
        Tz = Tz.clamp(min=self.v_min, max=self.v_max)
        
        # 投影到固定的原子网格上
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # 修正极端情况下的索引越界
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.n_atoms - 1)) * (l == u)] += 1

        # 分配概率
        target_probs = torch.zeros_like(current_probs)
        offset = torch.linspace(0, (len(rewards) - 1) * self.n_atoms, len(rewards)).long().unsqueeze(1).to(self.device)
        
        target_probs.view(-1).index_add_(0, (l + offset).view(-1), (next_probs * (u.float() - b)).view(-1))
        target_probs.view(-1).index_add_(0, (u + offset).view(-1), (next_probs * (b - l.float())).view(-1))

        # 4. 计算交叉熵损失
        loss = -(target_probs * current_probs.log()).sum(-1).mean()
        
        # 5. 优化和更新目标网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 重置 Noisy Net 的噪声
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        return loss.item()

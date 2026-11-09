"""IQN：Implicit Quantile Network。"""
import torch
import torch.nn.functional as F
from .base import BaseRLAlgo


class IQN(BaseRLAlgo):
    def __init__(self, obs_dim, action_dim, num_quantiles=32, kappa=1.0, gamma=0.99, **kwargs):
        super().__init__(**kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles  # 每次更新采样的分位数数量
        self.kappa = kappa                  # Huber 损失的阈值
        self.gamma = gamma                  # 折扣因子
        
        # 假设 BaseRLAlgo 中定义或可通过 kwargs 传入网络和优化器
        # self.q_net = QNetwork(obs_dim, action_dim, num_quantiles)
        # self.target_q_net = QNetwork(obs_dim, action_dim, num_quantiles)
        # self.optimizer = torch.optim.Adam(self.q_net.parameters())

    def select_action(self, obs):
        # 选择动作：通过 IQN 网络计算期望 Q 值，并选取 Q 值最大的动作
        with torch.no_grad():
            # 假设网络前向传播返回分位数值 和分位数
            # 为了得到期望 Q 值，我们对所有采样的分位数值求均值
            quantile_values, _ = self.q_net(obs)
            # quantile_values 形状通常为 (batch_size, num_quantiles, action_dim)
            expected_q_values = quantile_values.mean(dim=1)  # (batch_size, action_dim)
            action = expected_q_values.argmax(dim=1)
        return action

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch

        # 1. 采样分位数
        tau = torch.rand(self.num_quantiles, device=obs.device)
        tau_next = torch.rand(self.num_quantiles, device=obs.device)

        # 2. 计算当前状态的分位数值
        # 形状: (batch_size, num_quantiles, action_dim)
        current_quantiles, _ = self.q_net(obs, tau=tau)
        # 提取对应动作的分位数值，形状: (batch_size, num_quantiles)
        current_quantiles = current_quantiles.gather(2, actions.unsqueeze(1).unsqueeze(2).expand(-1, self.num_quantiles, 1)).squeeze(2)

        # 3. 计算目标分位数值
        with torch.no_grad():
            # 计算下一状态的期望 Q 值以选择下一动作 (Double IQN 思想)
            next_quantiles, _ = self.q_net(next_obs, tau=tau_next)
            next_q_values = next_quantiles.mean(dim=1)
            next_actions = next_q_values.argmax(dim=1)

            # 使用目标网络计算下一状态的目标分位数值
            target_next_quantiles, _ = self.target_q_net(next_obs, tau=tau_next)
            target_next_quantiles = target_next_quantiles.gather(2, next_actions.unsqueeze(1).unsqueeze(2).expand(-1, self.num_quantiles, 1)).squeeze(2)

            # 计算 TD 目标: R + gamma * Q_target(s', a')
            target_quantiles = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_next_quantiles

        # 4. 计算分位数 Huber 损失
        # 扩展维度以进行成对计算
        # current_quantiles: (batch_size, num_quantiles, 1)
        # target_quantiles: (batch_size, 1, num_quantiles)
        td_errors = target_quantiles.unsqueeze(1) - current_quantiles.unsqueeze(2)
        
        # 计算分位数 Huber 损失
        # tau: (1, num_quantiles, 1)
        indicator = (td_errors < 0).float()
        weight = torch.abs(tau.unsqueeze(0).unsqueeze(2) - indicator)
        
        # Huber 损失计算
        abs_td_errors = torch.abs(td_errors)
        quadratic = torch.min(abs_td_errors, torch.tensor(self.kappa, device=obs.device))
        linear = abs_td_errors - quadratic
        huber_loss = 0.5 * quadratic ** 2 + self.kappa * linear
        
        # 加权并求均值
        quantile_loss = (weight * huber_loss).mean()

        # 5. 优化模型
        self.optimizer.zero_grad()
        quantile_loss.backward()
        self.optimizer.step()

        # 更新目标网络 (假设 BaseRLAlgo 中有相关机制)
        if hasattr(self, 'target_update'):
            self.soft_update_target()

        return quantile_loss.item()

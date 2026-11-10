"""A2C：Advantage Actor-Critic（骨架）。"""
from .base import BaseRLAlgo
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class A2C(BaseRLAlgo):
    """Advantage Actor-Critic 算法实现。
    
    假设 policy_net 输出元组: (action_logits, state_value)
    - action_logits: 用于计算动作分布
    - state_value: V(s)
    """

    def __init__(self, config: dict, policy_net: nn.Module,
                 target_net=None, optimizer=None):
        # A2C 通常不需要单独的 target_net，Critic 使用 policy_net 输出的 V(s)
        super().__init__(config, policy_net, target_net=None)
        
        # 从配置中获取超参数
        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)
        
        # 优化器
        self.optimizer = optimizer or torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 1e-3))

    def act(self, obs, deterministic=False):
        """统一简洁接口。"""
        a, _, _ = self._forward(obs, deterministic=deterministic)
        if isinstance(a, torch.Tensor) and a.dim() == 0:
            return int(a.item())
        return a

    def act_with_detail(self, obs, deterministic=False):
        """返回 (action, log_prob, value)。"""
        return self._forward(obs, deterministic=deterministic)

    @torch.no_grad()
    def _forward(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 2:
            obs_t = obs_t.unsqueeze(0)
        logits, value = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        if obs_t.dim() == 3:
            return int(action.item()), float(log_prob.item()), float(value.squeeze(-1).item())
        return action, log_prob, value.squeeze(-1)

    def select_action(self, obs: torch.Tensor):
        return self.act_with_detail(obs)

    def update(self, batch: Dict) -> Dict[str, float]:
        """使用收集到的轨迹数据进行 A2C 更新。
        
        Args:
            batch: 包含 'log_probs', 'values', 'rewards', 'dones', 'next_values' 的字典
                   - log_probs: 旧的动作对数概率
                   - values: 旧的状态价值 V(s)
                   - rewards: 每一步的奖励
                   - dones: 每一步的终止标志
                   - next_values: 最后一步的下一个状态价值 V(s_{t+1})
                   
        Returns:
            包含各项损失值的字典
        """
        log_probs = batch['log_probs']
        values = batch['values']
        rewards = batch['rewards']
        dones = batch['dones']
        next_values = batch['next_values']

        # 计算回报: 使用 V(s_{t+1}) 作为自举项计算优势函数
        # R_t = r_t + gamma * V(s_{t+1}) * (1 - done)
        returns = rewards + self.gamma * next_values * (1.0 - dones)
        
        # 计算优势函数: A(s, a) = R_t - V(s)
        advantages = returns.detach() - values.detach()
        
        # 1. Actor 损失 (策略梯度损失): -log_prob * advantage
        actor_loss = -(log_probs * advantages).mean()
        
        # 2. Critic 损失: 均方误差
        critic_loss = F.mse_loss(values, returns.detach())
        
        # 3. 熵奖励: 鼓励策略探索
        # 需要重新计算分布以获取熵
        # 注意：如果为了极致性能，可以在 select_action 时保存 entropy，此处直接重新计算以保持接口简洁
        with torch.no_grad():
            action_logits, _ = self.policy(batch.get('obs', torch.tensor([])).to(self.device))
        if action_logits.numel() > 0:
            dist = torch.distributions.Categorical(logits=action_logits)
            entropy = dist.entropy().mean()
        else:
            entropy = torch.tensor(0.0, device=self.device)

        # 总损失 = Actor损失 + Value损失系数 * Critic损失 - 熵系数 * 熵
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
        
        # 梯度清零、反向传播、更新参数
        self.optimizer.zero_grad()
        loss.backward()
        # 可选：梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {
            "total_loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        }

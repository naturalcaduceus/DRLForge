"""REINFORCE：蒙特卡洛策略梯度
REINFORCE 是最基础的策略梯度算法，它通过采样完整的回合数据，利用从回合开始到结束的总回报来更新策略。

‌REINFORCE是一种基于策略梯度的强化学习算法‌，核心是直接优化策略本身，
而不是像Q-learning那样通过价值函数间接学习策略。‌‌
它属于‌无模型、蒙特卡洛策略梯度方法‌，由Ronald J. Williams在1992年提出。‌‌
核心原理
REINFORCE的思路很直观：让智能体与环境交互，跑完一整条轨迹后，‌如果这条轨迹的回报高，就增加轨迹中每个动作的概率；回报低或为负，就降低这些动作的概率‌。‌‌
具体流程是：
‌采样轨迹‌：用当前策略与环境交互，收集一条完整轨迹（状态、动作、奖励序列）。
‌计算回报‌：从后往前计算每个时间步的折扣累积奖励 
‌更新策略‌：用梯度上升更新策略参数，让高回报动作的概率变大。更新公式核心是
⚠️ 主要特点与局限
‌优点‌：原理简单，直接优化目标函数，天然支持连续动作空间，不需要环境模型。
‌缺点‌：‌梯度估计方差很大‌，因为单条轨迹的回报波动剧烈，导致训练不稳定、样本效率低。‌‌

"""
from .base import BaseRLAlgo

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class REINFORCE(BaseRLAlgo):
    """REINFORCE 算法实现。
    
    假设 policy_net 输出动作的 logits (适用于离散动作空间)。
    """

    def __init__(self, config: dict, policy_net: nn.Module, optimizer=None):
        # REINFORCE 只有策略网络，没有目标网络
        super().__init__(config, policy_net, target_net=None)
        
        # 超参数
        self.gamma = config.get("gamma", 0.99)
        self.entropy_coef = config.get("entropy_coef", 0.01)  # 熵正则化系数，鼓励探索
        
        # 优化器
        self.optimizer = optimizer or torch.optim.Adam(self.policy.parameters(), lr=config.get("lr", 1e-3))

    def act(self, obs, deterministic=False):
        """统一简洁接口（无梯度，评估用）。"""
        a, _ = self._forward(obs, deterministic=deterministic)
        if isinstance(a, torch.Tensor) and a.dim() == 0:
            return int(a.item())
        return a

    def act_with_detail(self, obs, deterministic=False):
        """返回 (action, log_prob)，带梯度，供轨迹采样用。"""
        return self._forward(obs, deterministic=deterministic)

    def _forward(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 2:
            obs_t = obs_t.unsqueeze(0)
        action_logits = self.policy(obs_t)
        dist = torch.distributions.Categorical(logits=action_logits)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def select_action(self, obs: torch.Tensor):
        return self.act_with_detail(obs)

    def update(self, batch: Dict) -> Dict[str, float]:
        """使用一个完整回合的轨迹数据进行 REINFORCE 更新。
        
        Args:
            batch: 包含一条完整轨迹的字典，需包含:
                   - 'log_probs': 每一步动作的对数概率列表/张量
                   - 'rewards': 每一步的奖励列表/张量
                   
        Returns:
            包含损失值的字典
        """
        lp_raw = batch['log_probs']
        log_probs = (torch.stack(lp_raw) if isinstance(lp_raw, list) else lp_raw).to(self.device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)
        
        # 1. 计算每一步的回报 G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
        returns = self._compute_returns(rewards)
        
        # 2. 策略梯度损失: -log_prob * G_t
        # 取负号是因为 PyTorch 默认最小化损失，而我们希望最大化目标函数
        policy_loss = -(log_probs * returns).mean()
        
        # 3. 计算熵奖励 (可选，用于鼓励探索，防止过早收敛)
        # 注意：如果为了节省计算，可以在 select_action 时保存 entropy
        # 这里重新计算以保持接口简洁
        obs = torch.stack(batch['obs']).to(self.device) if 'obs' in batch else None
        if obs is not None:
            with torch.no_grad():
                action_logits = self.policy(obs)
                dist = torch.distributions.Categorical(logits=action_logits)
                entropy = dist.entropy().mean()
        else:
            entropy = torch.tensor(0.0, device=self.device)
            
        entropy_loss = -self.entropy_coef * entropy
        
        # 总损失
        loss = policy_loss + entropy_loss
        
        # 梯度清零、反向传播、更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item()
        }

    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """从后往前迭代计算每一步的蒙特卡洛回报。
        
        Args:
            rewards: 形状为 (T,) 的奖励张量
            
        Returns:
            returns: 形状为 (T,) 的回报张量
        """
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        
        # 逆序迭代计算
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
            
        # 对回报进行标准化 (减去均值除以标准差) 是一种常见的方差缩减技术
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        return returns

"""TD3：Twin Delayed DDPG"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .base import BaseRLAlgo


class TD3(BaseRLAlgo):
    """Twin Delayed DDPG 算法实现。
    
    注意：在 BaseRLAlgo 中，self.policy 对应 actor，self.target 对应 target_actor。
    """

    def __init__(self, config: dict, actor: nn.Module, q1: nn.Module, q2: nn.Module,
                 target_actor: nn.Module = None, target_q1: nn.Module = None, target_q2: nn.Module = None,
                 actor_optimizer=None, critic_optimizer=None):
        # 如果没有提供目标网络，则自动深拷贝一份
        if target_actor is None:
            target_actor = copy.deepcopy(actor)
        if target_q1 is None:
            target_q1 = copy.deepcopy(q1)
        if target_q2 is None:
            target_q2 = copy.deepcopy(q2)
            
        super().__init__(config, actor, target_actor)
        
        self.q1 = q1.to(self.device)
        self.q2 = q2.to(self.device)
        self.target_q1 = target_q1.to(self.device)
        self.target_q2 = target_q2.to(self.device)
        
        # 超参数
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)           # 软更新系数
        self.policy_noise = config.get("policy_noise", 0.2)  # 目标策略平滑噪声标准差
        self.noise_clip = config.get("noise_clip", 0.5)      # 目标策略平滑噪声截断值
        self.policy_delay = config.get("policy_delay", 2)    # 策略延迟更新频率
        
        # 优化器 (TD3 通常让两个 Critic 共享一个优化器，以便同时进行梯度清零和更新)
        self.actor_optimizer = actor_optimizer or torch.optim.Adam(self.policy.parameters(), lr=config.get("actor_lr", 1e-4))
        critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        self.critic_optimizer = critic_optimizer or torch.optim.Adam(critic_params, lr=config.get("critic_lr", 1e-3))
        
        # 训练步数计数器，用于实现延迟更新
        self.total_it = 0
        
        # 初始化时，目标网络权重与主网络完全一致
        self.hard_update(self.target, self.policy)
        self.hard_update(self.target_q1, self.q1)
        self.hard_update(self.target_q2, self.q2)

    def select_action(self, obs: torch.Tensor, explore: bool = True) -> torch.Tensor:
        """根据观测值选择连续动作。
        
        Args:
            obs: 环境观测值
            explore: 是否添加探索噪声 (训练时通常为 True，测试时为 False)
            
        Returns:
            action: 连续动作张量
        """
        obs = obs.to(self.device)
        
        with torch.no_grad():
            action = self.policy(obs)
            
        if explore:
            # TD3 探索通常使用与 DDPG 类似的高斯噪声
            exploration_noise_std = self.cfg.get("exploration_noise", 0.1)
            noise = torch.randn_like(action) * exploration_noise_std
            action = action + noise
            
        # 假设动作空间为 [-1, 1]
        action = action.clamp(-1.0, 1.0)
        
        return action

    def update(self, batch: Dict) -> Dict[str, float]:
        """使用经验回放池的数据更新 TD3 网络。
        
        Args:
            batch: 包含 'obs', 'action', 'reward', 'next_obs', 'done' 的字典
                   
        Returns:
            包含各项损失值的字典
        """
        self.total_it += 1
        
        obs = batch['obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)

        # ---------------------- 更新 Critic ----------------------
        with torch.no_grad():
            # 1. 选择下一步动作：Target Actor
            next_action = self.target(next_obs)
            
            # 2. 目标策略平滑：给目标动作添加截断噪声
            noise = (torch.randn_like(next_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action_smooth = (next_action + noise).clamp(-1.0, 1.0)
            
            # 3. 计算目标 Q 值：取两个 Target Q 网络的最小值
            target_Q1 = self.target_q1(next_obs, next_action_smooth)
            target_Q2 = self.target_q2(next_obs, next_action_smooth)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # 4. 计算 TD 目标
            y = reward + self.gamma * (1.0 - done) * target_Q
            
        # 获取当前 Q 估计
        current_Q1 = self.q1(obs, action)
        current_Q2 = self.q2(obs, action)
        
        # 计算 Critic 损失
        critic_loss = F.mse_loss(current_Q1, y) + F.mse_loss(current_Q2, y)
        
        # 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0)
        self.critic_optimizer.step()

        # ---------------------- 延迟更新 Actor & 目标网络 ----------------------
        actor_loss_val = 0.0
        if self.total_it % self.policy_delay == 0:
            # 1. 计算 Actor 损失：仅使用 Q1 进行梯度回传 (也可以用 min(Q1, Q2)，但原论文实现多用 Q1)
            actor_action = self.policy(obs)
            actor_loss = -self.q1(obs, actor_action).mean()
            
            # 2. 优化 Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # 3. 软更新目标网络
            self.soft_update(self.target, self.policy, self.tau)
            self.soft_update(self.target_q1, self.q1, self.tau)
            self.soft_update(self.target_q2, self.q2, self.tau)
            
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_val
        }

    # ------- 辅助方法 -------

    @staticmethod
    def soft_update(target_net: nn.Module, main_net: nn.Module, tau: float):
        """Polyak 平均软更新"""
        for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(target_net: nn.Module, main_net: nn.Module):
        """硬更新"""
        target_net.load_state_dict(main_net.state_dict())

    # ------- 重写保存/加载以包含 Critic -------

    def save_checkpoint(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target": self.target.state_dict() if self.target else None,
            "target_q1": self.target_q1.state_dict() if self.target_q1 else None,
            "target_q2": self.target_q2.state_dict() if self.target_q2 else None,
            "cfg": self.cfg,
            "total_it": self.total_it,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        if self.target is not None and ckpt.get("target") is not None:
            self.target.load_state_dict(ckpt["target"])
        if self.target_q1 is not None and ckpt.get("target_q1") is not None:
            self.target_q1.load_state_dict(ckpt["target_q1"])
        if self.target_q2 is not None and ckpt.get("target_q2") is not None:
            self.target_q2.load_state_dict(ckpt["target_q2"])
        if "total_it" in ckpt:
            self.total_it = ckpt["total_it"]

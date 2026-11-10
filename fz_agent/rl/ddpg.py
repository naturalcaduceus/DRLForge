"""DDPG：深度确定性策略梯度"""
from .base import BaseRLAlgo
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DDPG(BaseRLAlgo):
    """Deep Deterministic Policy Gradient 算法实现。
    
    注意：在 BaseRLAlgo 中，self.policy 对应 actor，self.target 对应 target_actor。
    """

    def __init__(self, config: dict, actor: nn.Module, critic: nn.Module,
                 target_actor: nn.Module = None, target_critic: nn.Module = None,
                 actor_optimizer=None, critic_optimizer=None):
        # 如果没有提供目标网络，则自动深拷贝一份
        if target_actor is None:
            target_actor = copy.deepcopy(actor)
        if target_critic is None:
            target_critic = copy.deepcopy(critic)
            
        super().__init__(config, actor, target_actor)
        
        self.critic = critic.to(self.device)
        self.target_critic = target_critic.to(self.device)
        
        # 超参数
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)  # 软更新系数
        
        # 优化器
        self.actor_optimizer = actor_optimizer or torch.optim.Adam(self.policy.parameters(), lr=config.get("actor_lr", 1e-4))
        self.critic_optimizer = critic_optimizer or torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 1e-3))
        
        # 动作噪声 (用于探索)
        self.noise_std = config.get("noise_std", 0.1)
        self.noise_clip = config.get("noise_clip", 0.5)
        
        # 初始化时，目标网络权重与主网络完全一致
        self.hard_update(self.target, self.policy)
        self.hard_update(self.target_critic, self.critic)

    def act(self, obs, deterministic=False):
        """统一简洁接口：返回连续动作 tensor。"""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        return self.select_action(obs_t, explore=not deterministic)

    def select_action(self, obs: torch.Tensor, explore: bool = True) -> torch.Tensor:
        """根据观测值选择连续动作。
        
        Args:
            obs: 环境观测值
            explore: 是否添加噪声进行探索 (训练时通常为 True，测试时为 False)
            
        Returns:
            action: 连续动作张量
        """
        obs = obs.to(self.device)
        
        # Actor 输出确定性动作
        with torch.no_grad():
            action = self.policy(obs)
            
        if explore:
            # 添加高斯噪声进行探索 (也可替换为 OU Noise)
            noise = torch.randn_like(action) * self.noise_std
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            action = action + noise
            
        # 假设动作空间为 [-1, 1]，如果为其他范围需在此处做缩放
        action = action.clamp(-1.0, 1.0)
        
        return action

    def update(self, batch: Dict) -> Dict[str, float]:
        """使用经验回放池的数据更新 DDPG 网络。
        
        Args:
            batch: 包含 'obs', 'action', 'reward', 'next_obs', 'done' 的字典
                   
        Returns:
            包含各项损失值的字典
        """
        obs = batch['obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)

        # ---------------------- 更新 Critic ----------------------
        # 1. 计算目标 Q 值: y = r + gamma * Q_target(s', mu_target(s'))
        with torch.no_grad():
            next_action = self.target(next_obs)
            target_q_value = self.target_critic(next_obs, next_action)
            y = reward + self.gamma * target_q_value * (1.0 - done)
            
        # 2. 计算当前 Q 值
        current_q_value = self.critic(obs, action)
        
        # 3. 计算 Critic 损失 (均方误差)
        critic_loss = F.mse_loss(current_q_value, y)
        
        # 4. 优化 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ---------------------- 更新 Actor ----------------------
        # 1. 计算 Actor 损失: -Q(s, mu(s))，即最大化 Q 值
        # 注意：这里不能用 torch.no_grad()，因为需要对 Actor 的参数求梯度
        actor_action = self.policy(obs)
        actor_loss = -self.critic(obs, actor_action).mean()
        
        # 2. 优化 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ---------------------- 软更新目标网络 ----------------------
        self.soft_update(self.target, self.policy, self.tau)
        self.soft_update(self.target_critic, self.critic, self.tau)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item()
        }

    # ------- 辅助方法 -------

    @staticmethod
    def soft_update(target_net: nn.Module, main_net: nn.Module, tau: float):
        """Polyak 平均软更新: theta_target = tau * theta_main + (1 - tau) * theta_target"""
        for target_param, main_param in zip(target_net.parameters(), main_net.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    @staticmethod
    def hard_update(target_net: nn.Module, main_net: nn.Module):
        """硬更新: 直接复制权重"""
        target_net.load_state_dict(main_net.state_dict())

    # ------- 重写保存/加载以包含 Critic -------

    def save_checkpoint(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict() if self.target else None,
            "target_critic": self.target_critic.state_dict() if self.target_critic else None,
            "cfg": self.cfg,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        if self.target is not None and ckpt.get("target") is not None:
            self.target.load_state_dict(ckpt["target"])
        if self.target_critic is not None and ckpt.get("target_critic") is not None:
            self.target_critic.load_state_dict(ckpt["target_critic"])

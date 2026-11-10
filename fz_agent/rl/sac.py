import torch
import torch.nn.functional as F
from torch.distributions import Normal

"""SAC：Soft Actor-Critic（完整实现）。"""
from .base import BaseRLAlgo


class SAC(BaseRLAlgo):
    def __init__(self, config, actor, q1, q2, target_q1, target_q2):
        super().__init__(config, actor)
        self.q1 = q1
        self.q2 = q2
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        
        # 假设从config中获取超参数
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.alpha_lr = config.get("alpha_lr", 3e-4)
        self.target_entropy = config.get("target_entropy", -torch.prod(torch.Tensor(config.get("action_dim", 1))).item())
        
        # 自动温度调节参数
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
        
        # 优化器 (假设actor和q网络已有内部优化器或在此处统一初始化)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_lr", 3e-4))
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.get("critic_lr", 3e-4))
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.get("critic_lr", 3e-4))
        
        # 硬更新目标网络初始权重
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

    def act(self, obs, deterministic=False):
        """统一简洁接口：返回连续动作 tensor。"""
        a, _ = self.select_action(obs, deterministic=deterministic)
        return a

    def select_action(self, obs, deterministic=False):
        # 将观测值转换为张量
        if not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
        # 通过Actor网络输出动作的均值和标准差
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # 使用重参数化技巧进行采样
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        
        # 计算tanh变换后的对数概率
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        if deterministic:
            # 确定性策略直接输出tanh(mean)
            action = torch.tanh(mean)
            return action.detach().cpu().numpy()[0], None
            
        return action.detach().cpu().numpy()[0], (action, log_prob)

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        
        # 将数据转移到设备上
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(self.device)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        
        alpha = self.log_alpha.exp()

        # ---------------------- 更新 Q 网络 ----------------------
        with torch.no_grad():
            # 采样下一步的动作及其对数概率
            next_mean, next_log_std = self.actor(next_obs)
            next_std = next_log_std.exp()
            next_normal = Normal(next_mean, next_std)
            next_x_t = next_normal.rsample()
            next_y_t = torch.tanh(next_x_t)
            next_actions = next_y_t
            
            next_log_prob = next_normal.log_prob(next_x_t) - torch.log(1 - next_y_t.pow(2) + 1e-6)
            next_log_prob = next_log_prob.sum(dim=-1, keepdim=True)
            
            # 计算目标Q值 (Target Q = r + gamma * (min(Q1, Q2) - alpha * log_prob))
            q1_target = self.target_q1(next_obs, next_actions)
            q2_target = self.target_q2(next_obs, next_actions)
            min_q_target = torch.min(q1_target, q2_target)
            
            target_q = rewards + self.gamma * (1 - dones) * (min_q_target - alpha * next_log_prob)
            
        # 当前Q值估计
        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        
        # 计算Q网络损失
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)
        
        # 优化 Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        # 优化 Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ---------------------- 更新 Actor 网络 ----------------------
        # 重新采样当前状态的动作，以保持梯度流通
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        reparam_actions = y_t
        
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # 计算Actor的目标值 (期望最大化 Q - alpha * log_prob，因此loss取负)
        q1_reparam = self.q1(obs, reparam_actions)
        q2_reparam = self.q2(obs, reparam_actions)
        min_q_reparam = torch.min(q1_reparam, q2_reparam)
        
        actor_loss = (alpha * log_prob - min_q_reparam).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- 更新温度参数 Alpha ----------------------
        # 自动调节Alpha以逼近目标熵
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---------------------- 软更新目标网络 ----------------------
        for param, target_param in zip(self.q1.parameters(), self.target_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.q2.parameters(), self.target_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item()
        }

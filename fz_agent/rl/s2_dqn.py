import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from config import gopt
from algos import BaseRLAlgo


# ============================================================
# 3. Q-Network
# ============================================================

class QNetwork(nn.Module):
    """
    输入：[batch, max_detected, obs_dim]
    输出：Q values：[batch, action_dim]
    """

    def __init__(self, obs_dim=3, hidden_dim=128, action_dim=6):
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

        # 输出每个动作的 Q 值
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        # [B, N]：非全零行表示一个有效实体。
        valid_mask = torch.any(torch.abs(obs) > 1e-8, dim=-1)

        # [B, N, H]
        entity_features = self.entity_encoder(obs)

        # Masked mean pooling，避免填充行影响聚合结果。
        mask = valid_mask.unsqueeze(-1).to(entity_features.dtype)
        feature_sum = (entity_features * mask).sum(dim=1)
        valid_count = mask.sum(dim=1).clamp(min=1.0)
        pooled = feature_sum / valid_count

        # 无可见实体时 pooled 为零向量，网络仍可正常前向计算。
        shared_feature = self.shared(pooled)

        q_values = self.q_head(shared_feature)
        return q_values


# ============================================================
# 4. Experience Replay Buffer
# ============================================================

class ReplayBuffer:
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


# ============================================================
# 5. DQN Agent
# ============================================================

class DQNAgent(BaseRLAlgo):
    def __init__(
        self,
        model,
        device,
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.005,            # 目标网络软更新系数
        epsilon_start=1.0,   # 初始探索率
        epsilon_end=0.05,    # 最低探索率
        epsilon_decay=0.995, # 探索率衰减因子
        max_grad_norm=10.0,
    ):
        self.model = model.to(device)
        self.target_model = QNetwork(
            obs_dim=model.entity_encoder[0].in_features,
            hidden_dim=model.shared[0].in_features,
            action_dim=model.q_head.out_features
        ).to(device)
        
        # 硬拷贝初始权重
        self.target_model.load_state_dict(self.model.state_dict())
        for param in self.target_model.parameters():
            param.requires_grad = False

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        if not deterministic:
            # Epsilon-Greedy 探索
            if random.random() < self.epsilon:
                action_dim = self.model.q_head.out_features
                return random.randint(0, action_dim - 1)

        obs_tensor = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        )
        if obs_tensor.dim() == 2:
            obs_tensor = obs_tensor.unsqueeze(0)

        q_values = self.model(obs_tensor)
        action = torch.argmax(q_values, dim=-1)
        return int(action.item())

    def update(self, replay_buffer, batch_size=64):
        if len(replay_buffer) < batch_size:
            return None

        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # 1. 获取当前 Q 值: Q(s, a)
        q_values = self.model(obs).gather(1, actions).squeeze(-1)

        # 2. 获取目标 Q 值: r + gamma * max(Q_target(s', a')) * (1 - done)
        with torch.no_grad():
            next_q_values = self.target_model(next_obs)
            max_next_q_values = next_q_values.max(dim=-1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1.0 - dones)

        # 3. 计算 Huber Loss (对 DQN 更稳定)
        loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

        # 4. 优化更新
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # 5. 软更新目标网络
        self.soft_update_target_network()

        return {"q_loss": loss.item(), "mean_q_value": q_values.mean().item()}

    def soft_update_target_network(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        """每次调用时衰减 Epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================
# 实例化
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = QNetwork(
    obs_dim=3,
    hidden_dim=128,
    action_dim=6,      # 行为空间维度
)

agent = DQNAgent(
    model=model,
    device=device,
    learning_rate=1e-3,
    gamma=0.99,
    tau=0.005,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    max_grad_norm=10.0,
)

replay_buffer = ReplayBuffer(capacity=10000)


# ============================================================
# 调用示例
# ============================================================
# # 1. 模拟环境交互收集数据
# num_steps = 100
# batch_size = 32
# 
# obs = np.random.randn(5, 3).astype(np.float32)  # 初始观测 [N, 3]
# 
# for step in range(num_steps):
#     # a. 使用 agent 选择动作 (包含 Epsilon-Greedy 探索)
#     action = agent.act(obs, deterministic=False)
#     
#     # b. 环境推进 (此处为模拟)
#     next_obs = np.random.randn(5, 3).astype(np.float32)
#     reward = np.random.randn()
#     done = random.choice([True, False]) 
#     
#     # c. 存入经验回放池
#     replay_buffer.push(obs, action, reward, next_obs, float(done))
#     
#     obs = next_obs if not done else np.random.randn(5, 3).astype(np.float32)
# 
#     # d. 当回放池数据足够时，进行网络更新
#     if len(replay_buffer) >= batch_size:
#         metrics = agent.update(replay_buffer, batch_size=batch_size)
#         if metrics is not None and step % 20 == 0:
#             print(f"Step {step}, Q Loss: {metrics['q_loss']:.4f}, Mean Q: {metrics['mean_q_value']:.4f}, Eps: {agent.epsilon:.3f}")
# 
# # 2. 每个 Episode 结束后衰减探索率
# agent.decay_epsilon()
# 
# # 3. 测试阶段的推理 (关闭探索，选择 Q 值最大的动作)
# test_obs = np.random.randn(5, 3).astype(np.float32)
# test_action = agent.act(test_obs, deterministic=True)
# print("测试阶段选择的动作:", test_action)

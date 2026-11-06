"""端到端 PPO 训练入口。

直接运行：
    cd /workspace/fz_agent && python main.py

或者从项目根：
    python -m fz_agent.main

功能：
    1. 构建简易对抗环境 CombatEnv
    2. 实例化 PPO + ActorCritic
    3. 收集 rollout → 更新 → 评估
    4. 定期打印训练指标和评估结果
"""
import os
import sys
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

# ------- 路径设置（保证 python main.py 和 python -m fz_agent.main 都能跑）-------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fz_agent.CombatEnv import CombatEnv
from fz_agent.rl.ppo import PPO, ActorCritic


# ============================================================
# 随机种子 & 设备
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Rollout 收集
# ============================================================

def collect_rollout(env, agent, obs, rollout_steps=1024):
    """收集一段 rollout，保证 episode 边界正确。"""
    buffer = {
        "obs": [], "actions": [], "log_probs": [],
        "rewards": [], "values": [], "next_values": [],
        "terminated": [], "episode_ends": [],
    }

    episode_rewards = []
    current_episode_reward = 0.0

    for _ in range(rollout_steps):
        action, log_prob, value = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_end = terminated or truncated

        next_value = 0.0 if terminated else agent.get_value(next_obs)

        buffer["obs"].append(np.asarray(obs, dtype=np.float32).copy())
        buffer["actions"].append(action)
        buffer["log_probs"].append(log_prob)
        buffer["rewards"].append(float(reward))
        buffer["values"].append(value)
        buffer["next_values"].append(next_value)
        buffer["terminated"].append(float(terminated))
        buffer["episode_ends"].append(float(episode_end))

        current_episode_reward += reward
        obs = next_obs

        if episode_end:
            episode_rewards.append(current_episode_reward)
            current_episode_reward = 0.0
            obs, _ = env.reset()

    return buffer, obs, episode_rewards, current_episode_reward


# ============================================================
# 评估
# ============================================================

def evaluate(env, agent, episodes=5):
    rewards, lengths = [], []
    wins = 0

    for _ in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        result = None

        while True:
            action, _, _ = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            result = info.get("result", result)
            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(steps)
        if result == "all_enemies_eliminated":
            wins += 1

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_length": float(np.mean(lengths)),
        "win_rate": wins / episodes,
    }


def save_checkpoint(agent, epoch, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": agent.policy.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }, path)


# ============================================================
# 主训练
# ============================================================

def main():
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # ------- 1. 环境 -------
    env = CombatEnv(
        num_enemies=3,
        max_detected=10,
        max_episode_steps=300,
        seed=42,
    )
    eval_env = CombatEnv(
        num_enemies=3,
        max_detected=10,
        max_episode_steps=300,
        seed=2026,
    )

    # ------- 2. 网络 + Agent -------
    obs_dim = 3          # CombatEnv 每个实体 3 维特征
    hidden_dim = 128
    action_dim = 3       # CombatEnv: 攻击 / 靠近 / 远离

    model = ActorCritic(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )

    agent = PPO(
        config={
            "device": str(device),
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        policy_net=model,
    )

    # ------- 3. 训练循环 -------
    total_epochs = 50
    rollout_steps = 1024
    update_epochs = 4
    minibatch_size = 128
    eval_interval = 5

    recent_rewards = deque(maxlen=20)
    obs, _ = env.reset()

    for epoch in range(1, total_epochs + 1):
        rollout, obs, completed_rewards, partial_reward = collect_rollout(
            env=env, agent=agent, obs=obs, rollout_steps=rollout_steps,
        )

        train_metrics = agent.update(
            rollout,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        )

        recent_rewards.extend(completed_rewards)

        if epoch % eval_interval == 0 or epoch == 1:
            eval_metrics = evaluate(eval_env, agent, episodes=5)

            avg_train_reward = (
                float(np.mean(recent_rewards)) if recent_rewards else float("nan")
            )

            print(
                f"[Epoch {epoch:03d}/{total_epochs}] "
                f"PolicyLoss={train_metrics['policy_loss']:.4f} "
                f"ValueLoss={train_metrics['value_loss']:.4f} "
                f"Entropy={train_metrics['entropy']:.4f} "
                f"KL={train_metrics['approx_kl']:.5f} "
                f"ClipFrac={train_metrics['clip_fraction']:.3f} | "
                f"TrainReward20={avg_train_reward:7.2f} "
                f"EvalReward={eval_metrics['mean_reward']:7.2f}±{eval_metrics['std_reward']:.2f} "
                f"EvalWinRate={eval_metrics['win_rate']:.1%} "
                f"EvalLen={eval_metrics['mean_length']:5.1f}"
            )

        if epoch % 20 == 0:
            ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = str(ckpt_dir / f"combat_ppo_epoch_{epoch}.pth")
            save_checkpoint(agent, epoch, ckpt_path)
            print(f"  ↳ Checkpoint saved: {ckpt_path}")

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    save_checkpoint(agent, total_epochs, str(ckpt_dir / "combat_ppo_final.pth"))
    print("\n训练完成 ✓")


if __name__ == "__main__":
    main()


# train_combat_ppo.py
# -*- coding: utf-8 -*-
"""
一个自包含、可运行的 PPO 示例。
依赖：numpy、torch
运行：python train_combat_ppo.py

说明：
1. CombatEnv 只是演示环境，接入真实仿真时替换该类即可。
2. 观测格式为 [max_detected, 3]，每行分别是相对 dx、相对 dy、敌方血量比例。
3. 使用实体 MLP + masked mean pooling，避免对全零填充实体进行无效聚合。
4. GAE 区分真正终止（terminated）和时间截断（truncated）。
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


from CombatEnv import CombatEnv

from config import gopt
# 引入各个模块
from fz_agent.rl.s1_ppo import PPOAgent


# ============================================================
# 1. 随机种子与设备
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
# 5. Rollout 收集
# ============================================================

def collect_rollout(env, agent, obs, rollout_steps=1024):
    """
    obs 由外部维护，保证相邻 Rollout 之间环境状态连续。
    对每个 transition 单独保存 next_value，正确处理 Episode 边界。
    """
    buffer = {
        "obs": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "values": [],
        "next_values": [],
        "terminated": [],
        "episode_ends": [],
    }

    episode_rewards = []
    current_episode_reward = 0.0

    for _ in range(rollout_steps):
        action, log_prob, value = agent.act(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_end = terminated or truncated

        # 必须使用本次 transition 的真实 next_obs 计算 Bootstrap value。
        # 即使 truncated，也要用截断前的 next_obs，而不是 reset 后的 obs。
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
# 6. 评估与保存
# ============================================================

def evaluate(env, agent, episodes=5):
    rewards = []
    lengths = []
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
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
        },
        path,
    )



# ============================================================
# 7. 主训练流程
# ============================================================

def main():
    set_seed(42)
    device = get_device()

    print(f"Device: {device}")


    # 输入输出
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



    total_epochs = 100
    rollout_steps = 1024
    update_epochs = 4
    minibatch_size = 128
    eval_interval = 5

    recent_rewards = deque(maxlen=20)

    obs, _ = env.reset()

    for epoch in range(1, total_epochs + 1):
        rollout, obs, completed_rewards, partial_reward = collect_rollout(
            env=env,
            agent=agent,
            obs=obs,
            rollout_steps=rollout_steps,
        )

        train_metrics = agent.update(
            rollout,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        )

        recent_rewards.extend(completed_rewards)

        if epoch % eval_interval == 0 or epoch == 1:
            eval_metrics = evaluate(
                eval_env,
                agent,
                episodes=5,
            )

            avg_train_reward = (
                float(np.mean(recent_rewards))
                if recent_rewards
                else float("nan")
            )

            print(
                f"[Epoch {epoch:03d}/{total_epochs}] "
                f"PolicyLoss={train_metrics['policy_loss']:.4f} "
                f"ValueLoss={train_metrics['value_loss']:.4f} "
                f"Entropy={train_metrics['entropy']:.4f} "
                f"KL={train_metrics['approx_kl']:.5f} "
                f"ClipFrac={train_metrics['clip_fraction']:.3f} | "
                f"TrainReward20={avg_train_reward:.2f} "
                f"EvalReward={eval_metrics['mean_reward']:.2f}±"
                f"{eval_metrics['std_reward']:.2f} "
                f"EvalWinRate={eval_metrics['win_rate']:.1%} "
                f"EvalLen={eval_metrics['mean_length']:.1f}"
            )

        if epoch % 20 == 0:
            checkpoint_path = f"checkpoints/combat_ppo_epoch_{epoch}.pth"
            save_checkpoint(agent, epoch, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    save_checkpoint(agent, total_epochs, "checkpoints/combat_ppo_final.pth")
    print("Training finished.")


if __name__ == "__main__":
    main()

"""端到端 PPO 训练入口。

直接运行：
    cd /workspace/fz_agent && python main.py

或者从项目根：
    python -m fz_agent.main

功能：
    1. 构建简易对抗环境 CombatEnv
    2. 实例化 PPO + ActorCritic
    3. 收集 rollout → 更新 → 评估
    4. 全量记录训练过程到 fz_agent/runs/<run_name>_<时间戳>/
       - config.json       配置快照
       - train_epoch.csv   每 epoch 训练指标
       - train_episode.csv 每 episode 回报/长度/结果
       - eval.csv          评估指标
       - events.jsonl      结构化事件流
       - train.log         文本日志
       - curves.png        训练曲线
       - tensorboard/      TensorBoard 日志
       - summary.json      结果汇总
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
from fz_agent.utils.logger import TrainLogger


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

def collect_rollout(env, agent, obs, rollout_steps=1024, env_steps_start=0):
    """收集一段 rollout，保证 episode 边界正确。

    返回 (buffer, next_obs, episode_records)，
    episode_records 每项包含 reward / length / result / env_steps。
    """
    buffer = {
        "obs": [], "actions": [], "log_probs": [],
        "rewards": [], "values": [], "next_values": [],
        "terminated": [], "episode_ends": [],
    }

    episode_records = []
    current_episode_reward = 0.0
    current_episode_steps = 0

    for i in range(rollout_steps):
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
        current_episode_steps += 1
        obs = next_obs

        if episode_end:
            episode_records.append({
                "reward": current_episode_reward,
                "length": current_episode_steps,
                "result": info.get("result", "unknown"),
                "env_steps": env_steps_start + i + 1,
            })
            current_episode_reward = 0.0
            current_episode_steps = 0
            obs, _ = env.reset()

    return buffer, obs, episode_records


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

    # ------- 0. 训练记录器 -------
    logger = TrainLogger(
        run_name="ppo_combat",
        use_tensorboard=True,
        plot=True,
    )
    logger.info("Device: %s", device)

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

    ppo_cfg = {
        "device": str(device),
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    agent = PPO(config=ppo_cfg, policy_net=model)

    # ------- 3. 训练循环 -------
    total_epochs = 50
    rollout_steps = 1024
    update_epochs = 4
    minibatch_size = 128
    eval_interval = 5

    logger.log_config({
        "env": {
            "name": "CombatEnv",
            "num_enemies": env.num_enemies,
            "map_size": env.map_size,
            "detect_range": env.detect_range,
            "max_detected": env.max_detected,
            "max_episode_steps": env.max_episode_steps,
        },
        "network": {
            "type": "ActorCritic",
            "obs_dim": obs_dim,
            "hidden_dim": hidden_dim,
            "action_dim": action_dim,
        },
        "algo": {"name": "ppo", "config": ppo_cfg},
        "train": {
            "total_epochs": total_epochs,
            "rollout_steps": rollout_steps,
            "update_epochs": update_epochs,
            "minibatch_size": minibatch_size,
            "eval_interval": eval_interval,
            "seed": 42,
        },
    })

    recent_rewards = deque(maxlen=20)
    obs, _ = env.reset()
    env_steps = 0

    for epoch in range(1, total_epochs + 1):
        rollout, obs, episode_records = collect_rollout(
            env=env, agent=agent, obs=obs, rollout_steps=rollout_steps,
            env_steps_start=env_steps,
        )
        env_steps += rollout_steps

        # 记录本段 rollout 中结束的每个 episode
        for rec in episode_records:
            logger.log_episode(
                reward=rec["reward"],
                length=rec["length"],
                result=rec["result"],
                env_steps=rec["env_steps"],
            )
            recent_rewards.append(rec["reward"])

        train_metrics = agent.update(
            rollout,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
        )

        avg_train_reward = (
            float(np.mean(recent_rewards)) if recent_rewards else float("nan")
        )

        logger.log_epoch(
            epoch=epoch,
            metrics=train_metrics,
            extra={
                "env_steps": env_steps,
                "train_reward_mean20": avg_train_reward,
                "episodes_in_rollout": len(episode_records),
            },
        )

        if epoch % eval_interval == 0 or epoch == 1:
            eval_metrics = evaluate(eval_env, agent, episodes=5)
            logger.log_eval(epoch, eval_metrics)

            logger.info(
                "[Epoch %03d/%03d] PolicyLoss=%.4f ValueLoss=%.4f "
                "Entropy=%.4f KL=%.5f ClipFrac=%.3f | "
                "TrainReward20=%7.2f EvalReward=%7.2f±%.2f "
                "EvalWinRate=%.1f%% EvalLen=%5.1f",
                epoch, total_epochs,
                train_metrics["policy_loss"], train_metrics["value_loss"],
                train_metrics["entropy"], train_metrics["approx_kl"],
                train_metrics["clip_fraction"],
                avg_train_reward,
                eval_metrics["mean_reward"], eval_metrics["std_reward"],
                eval_metrics["win_rate"] * 100,
                eval_metrics["mean_length"],
            )
        else:
            logger.info(
                "[Epoch %03d/%03d] PolicyLoss=%.4f ValueLoss=%.4f "
                "Entropy=%.4f TrainReward20=%7.2f",
                epoch, total_epochs,
                train_metrics["policy_loss"], train_metrics["value_loss"],
                train_metrics["entropy"], avg_train_reward,
            )

        if epoch % 20 == 0:
            ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = str(ckpt_dir / f"combat_ppo_epoch_{epoch}.pth")
            save_checkpoint(agent, epoch, ckpt_path)
            logger.log_event("checkpoint_saved", epoch=epoch, path=ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    final_path = str(ckpt_dir / "combat_ppo_final.pth")
    save_checkpoint(agent, total_epochs, final_path)
    logger.log_event("checkpoint_saved", epoch=total_epochs, path=final_path)

    logger.close(extra_summary={"checkpoint": final_path})
    print(f"\n训练完成 ✓  记录目录: {logger.run_dir}")


if __name__ == "__main__":
    main()

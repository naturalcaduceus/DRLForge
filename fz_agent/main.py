"""通用 RL 训练入口：从 JSON 配置启动训练。

用法：
    cd /workspace/fz_agent && python main.py
    cd /workspace/fz_agent && python main.py --config configs/ppo_combat.json
    cd /workspace && python -m fz_agent.main --config docs/1.json

训练循环按算法类型自动分派：
    - PPO / A2C       → on-policy rollout → GAE update
    - REINFORCE       → monte carlo 完整轨迹 → update
    - DQN / DDQN / DuelingDQN / Rainbow → replay buffer 采样
    - DDPG / TD3 / SAC → replay buffer + 两阶段 actor/critic 更新

日志记录、评估、检查点全部统一。
"""
import argparse
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fz_agent.CombatEnv import CombatEnv
from fz_agent.dl.factory import build_rl_from_json_config, load_json_config
from fz_agent.rl.dqn import ReplayBuffer
from fz_agent.utils.logger import TrainLogger


# ============================================================
# 工具
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(request: str = "cuda"):
    if request == "cuda" and not torch.cuda.is_available():
        print("[main] CUDA 不可用，自动回退 CPU")
        return torch.device("cpu")
    return torch.device(request)


def make_env(env_cfg: dict, seed: int = 42):
    env_type = env_cfg.get("type", "combat")
    if env_type == "combat":
        params = env_cfg.get("config", {})
        params.setdefault("num_enemies", 3)
        params.setdefault("max_detected", 10)
        params.setdefault("max_episode_steps", 300)
        params["seed"] = seed
        return CombatEnv(**params)
    raise ValueError(f"未知 env type: {env_type}（当前仅实现 combat）")


@torch.no_grad()
def evaluate(env, agent, episodes=5):
    rewards, lengths = [], []
    wins = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        total, steps, result = 0.0, 0, None
        while True:
            action = agent.act(obs, deterministic=True)
            if isinstance(action, torch.Tensor):
                if action.dim() == 0:
                    action = int(action.item())
                else:
                    action = action.cpu().numpy()
            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception:
                break
            total += reward
            steps += 1
            result = info.get("result", result) if isinstance(info, dict) else None
            if terminated or truncated:
                break
        rewards.append(total)
        lengths.append(steps)
        if result == "all_enemies_eliminated":
            wins += 1
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "win_rate": wins / max(1, len(rewards)),
    }


def save_checkpoint(agent, epoch, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {"policy": agent.policy.state_dict()}
    for attr in ("target", "critic", "q1", "q2", "target_q1", "target_q2",
                 "target_actor", "target_critic"):
        net = getattr(agent, attr, None)
        if net is not None:
            state[attr] = net.state_dict()
    for attr in ("optimizer", "actor_optimizer", "critic_optimizer",
                 "q1_optimizer", "q2_optimizer"):
        opt = getattr(agent, attr, None)
        if opt is not None:
            state[attr] = opt.state_dict()
    torch.save(state, path)


# ============================================================
# 采样函数（按算法类型）
# ============================================================

def collect_rollout_on_policy(env, agent, obs, rollout_steps=1024,
                              env_steps_start=0):
    """PPO / A2C 风格 rollout 收集。"""
    buffer = {
        "obs": [], "actions": [], "log_probs": [],
        "rewards": [], "values": [], "next_values": [],
        "terminated": [], "episode_ends": [],
    }
    ep_records = []
    cur_rew, cur_steps = 0.0, 0

    for i in range(rollout_steps):
        # PPO/A2C 带 act_with_detail，其他退化
        if hasattr(agent, "act_with_detail"):
            out = agent.act_with_detail(obs, deterministic=False)
            if len(out) == 3:
                action, log_prob, value = out
            else:
                action, log_prob = out
                value = 0.0
        else:
            action = agent.act(obs, deterministic=False)
            log_prob, value = 0.0, 0.0

        if isinstance(action, torch.Tensor):
            action = int(action.item()) if action.dim() == 0 else action.squeeze(0)
        if isinstance(log_prob, torch.Tensor):
            log_prob = log_prob.detach().cpu().numpy()
        if isinstance(value, torch.Tensor):
            value = float(value)

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_end = terminated or truncated
        next_value = 0.0 if terminated else (agent.get_value(next_obs)
                                             if hasattr(agent, "get_value") else 0.0)

        buffer["obs"].append(np.asarray(obs, dtype=np.float32).copy())
        buffer["actions"].append(action if isinstance(action, (int, np.integer)) else int(action))
        buffer["log_probs"].append(log_prob)
        buffer["rewards"].append(float(reward))
        buffer["values"].append(value if isinstance(value, (int, float)) else float(value))
        buffer["next_values"].append(next_value)
        buffer["terminated"].append(float(terminated))
        buffer["episode_ends"].append(float(episode_end))

        cur_rew += reward
        cur_steps += 1
        obs = next_obs

        if episode_end:
            ep_records.append({
                "reward": cur_rew, "length": cur_steps,
                "result": info.get("result", "unknown") if isinstance(info, dict) else "unknown",
                "env_steps": env_steps_start + i + 1,
            })
            cur_rew, cur_steps = 0.0, 0
            obs, _ = env.reset()

    return buffer, obs, ep_records


def collect_monte_carlo(env, agent, max_episodes=5):
    """REINFORCE：采样一整条轨迹后再 update。"""
    trajectories = []
    for _ in range(max_episodes):
        traj = {"obs": [], "log_probs": [], "rewards": []}
        obs, _ = env.reset()
        done = False
        while not done:
            if hasattr(agent, "act_with_detail"):
                a, lp = agent.act_with_detail(obs, deterministic=False)
            else:
                a, lp = agent.act(obs, deterministic=False), None
            if isinstance(a, torch.Tensor):
                a = int(a.item()) if a.dim() == 0 else a.squeeze(0)

            traj["obs"].append(np.asarray(obs, dtype=np.float32).copy())
            traj["log_probs"].append(lp)  # 保持 tensor（带 grad）

            obs, r, terminated, truncated, _ = env.step(int(a))
            traj["rewards"].append(float(r))
            done = terminated or truncated
        trajectories.append(traj)
    return trajectories


def collect_replay_step(env, agent, obs, rb: ReplayBuffer, deterministic=False):
    """DQN 家族 / 连续动作：单 step 经验存入 replay buffer。"""
    action = agent.act(obs, deterministic=deterministic)
    if isinstance(action, torch.Tensor):
        if action.dim() == 0:
            action = int(action.item())
        else:
            action = action.cpu().numpy().flatten()

    next_obs, reward, terminated, truncated, _ = env.step(action)
    rb.push(obs, action, reward, next_obs, terminated or truncated)
    return next_obs, terminated or truncated, reward


# ============================================================
# 训练循环（按算法分派）
# ============================================================

def train_discrete_ppo_like(env, eval_env, agent, cfg, logger, algo_name):
    """PPO / A2C 这类 on-policy 离散动作算法。"""
    algo_cfg = cfg["algo"]["config"]
    train_cfg = cfg.get("train", {})

    total_epochs = train_cfg.get("total_epochs", 50)
    rollout_steps = train_cfg.get("rollout_steps", 1024)
    update_epochs = train_cfg.get("update_epochs", algo_cfg.get("ppo_epochs", algo_cfg.get("n_epochs", 4)))
    minibatch_size = train_cfg.get("minibatch_size", algo_cfg.get("batch_size", 128))
    eval_interval = train_cfg.get("eval_interval", 5)

    recent = deque(maxlen=20)
    obs, _ = env.reset()
    env_steps = 0

    for epoch in range(1, total_epochs + 1):
        rollout, obs, ep_records = collect_rollout_on_policy(
            env, agent, obs, rollout_steps=rollout_steps, env_steps_start=env_steps,
        )
        env_steps += rollout_steps
        for r in ep_records:
            logger.log_episode(**{k: r[k] for k in ("reward", "length", "result", "env_steps")})
            recent.append(r["reward"])

        if algo_name == "ppo":
            metrics = agent.update(
                rollout, update_epochs=update_epochs, minibatch_size=minibatch_size,
            )
        elif algo_name == "a2c":
            batch = {k: torch.tensor(v, dtype=torch.float32, device=agent.device)
                     for k, v in rollout.items() if k != "obs"}
            batch["log_probs"] = batch["log_probs"].unsqueeze(-1)
            metrics = agent.update(batch) or {"loss": 0.0}
        else:
            metrics = {"loss": 0.0}

        avg = float(np.mean(recent)) if recent else float("nan")
        logger.log_epoch(epoch=epoch, metrics=metrics, extra={
            "env_steps": env_steps, "train_reward_mean20": avg,
        })

        if epoch % eval_interval == 0 or epoch == 1:
            em = evaluate(eval_env, agent, episodes=5)
            logger.log_eval(epoch, em)
            _log_train_info(logger, algo_name, epoch, total_epochs, metrics, em, avg)
        else:
            logger.info("[%s] Epoch %03d | reward20=%.2f | %s",
                        algo_name.upper(), epoch, avg,
                        " ".join(_fmt_metrics(metrics))[:60])

        if epoch % max(1, total_epochs // 5) == 0:
            save_checkpoint(agent, epoch,
                            str(_THIS_DIR / "checkpoints" / f"{algo_name}_epoch_{epoch}.pth"))

    save_checkpoint(agent, total_epochs,
                    str(_THIS_DIR / "checkpoints" / f"{algo_name}_final.pth"))
    logger.close(extra_summary={"algo": algo_name})


def train_reinforce(env, eval_env, agent, cfg, logger):
    algo_cfg = cfg["algo"]["config"]
    train_cfg = cfg.get("train", {})
    total_epochs = train_cfg.get("total_epochs", 50)
    ep_per_epoch = train_cfg.get("episodes_per_epoch", 5)
    eval_interval = train_cfg.get("eval_interval", 5)

    recent = deque(maxlen=20)
    for epoch in range(1, total_epochs + 1):
        trajs = collect_monte_carlo(env, agent, max_episodes=ep_per_epoch)

        all_metrics = {}
        for t in trajs:
            lp_list = t["log_probs"]
            if isinstance(lp_list[0], torch.Tensor):
                lp_tensor = torch.stack(lp_list).to(agent.device)
            else:
                lp_tensor = torch.as_tensor(lp_list, dtype=torch.float32, device=agent.device)
            batch = {
                "log_probs": lp_tensor,
                "rewards": t["rewards"],
            }
            try:
                m = agent.update(batch) or {"loss": 0.0}
                for k, v in m.items():
                    all_metrics.setdefault(k, []).append(v)
            except Exception as e:
                all_metrics.setdefault("error", []).append(str(e)[:80])

            R = sum(t["rewards"])
            recent.append(R)
            logger.log_episode(reward=R, length=len(t["rewards"]),
                               result="rollout",
                               env_steps=epoch * len(t["rewards"]))

        metrics = {k: float(np.mean(v)) if all(isinstance(x, (int, float)) for x in v) else str(v[0])
                   for k, v in all_metrics.items()}
        avg = float(np.mean(recent)) if recent else 0.0
        logger.log_epoch(epoch=epoch, metrics=metrics, extra={"train_reward_mean20": avg})

        if epoch % eval_interval == 0 or epoch == 1:
            em = evaluate(eval_env, agent, episodes=5)
            logger.log_eval(epoch, em)
            _log_train_info(logger, "reinforce", epoch, total_epochs, metrics, em, avg)
        else:
            logger.info("[REINFORCE] Epoch %03d | reward20=%.2f | %s",
                        epoch, avg, " ".join(_fmt_metrics(metrics))[:60])

    save_checkpoint(agent, total_epochs, str(_THIS_DIR / "checkpoints" / "reinforce_final.pth"))
    logger.close(extra_summary={"algo": "reinforce"})


def train_dqn_like(env, eval_env, agent, cfg, logger, algo_name):
    """DQN / DDQN / DuelingDQN / Rainbow。"""
    algo_cfg = cfg["algo"]["config"]
    train_cfg = cfg.get("train", {})

    total_steps = train_cfg.get("total_steps", 20000)
    eval_interval = train_cfg.get("eval_interval_steps", 2000)
    rb_capacity = train_cfg.get("capacity", algo_cfg.get("capacity", 10000))
    warmup_steps = train_cfg.get("warmup_steps", 1000)

    obs_shape = env.reset()[0].shape
    act_dim = env.action_space.n if hasattr(env, "action_space") else algo_cfg.get("action_dim", 3)
    rb = ReplayBuffer(capacity=rb_capacity)

    recent = deque(maxlen=20)
    obs, _ = env.reset()
    done = False
    cur_rew, cur_steps = 0.0, 0

    for step in range(1, total_steps + 1):
        # 探索期 / epsilon-greedy
        explore = step <= warmup_steps or (
            hasattr(agent, "epsilon") and random.random() < agent.epsilon
        )
        action = agent.act(obs, deterministic=not explore)
        if isinstance(action, torch.Tensor):
            action = int(action.item()) if action.dim() == 0 else int(action.squeeze())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        rb.push(obs, action, float(reward), next_obs, terminated or truncated)

        cur_rew += reward
        cur_steps += 1
        obs = next_obs

        # update
        try:
            metrics = agent.update(rb)
        except Exception:
            metrics = None

        if terminated or truncated:
            recent.append(cur_rew)
            logger.log_episode(reward=cur_rew, length=cur_steps,
                               result="ended", env_steps=step)
            cur_rew, cur_steps = 0.0, 0
            obs, _ = env.reset()

        if step % eval_interval == 0:
            em = evaluate(eval_env, agent, episodes=5)
            avg = float(np.mean(recent)) if recent else 0.0
            logger.log_eval(step, em)
            logger.info("[%s] step %05d/%05d | eval=%.2f win=%.0f%% | reward20=%.2f",
                        algo_name.upper(), step, total_steps,
                        em["mean_reward"], em["win_rate"] * 100, avg)

    save_checkpoint(agent, total_steps, str(_THIS_DIR / "checkpoints" / f"{algo_name}_final.pth"))
    logger.close(extra_summary={"algo": algo_name})


def train_continuous(env, eval_env, agent, cfg, logger, algo_name):
    """DDPG / TD3 / SAC。预留：环境需要连续动作空间。"""
    logger.info("[%s] 连续动作训练循环需要连续动作环境，暂未接入 CombatEnv", algo_name.upper())
    logger.close()


# ============================================================
# 日志辅助
# ============================================================

def _fmt_metrics(metrics: dict) -> list:
    out = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out.append(f"{k}={v:.4f}")
        else:
            out.append(f"{k}={str(v)[:20]}")
    return out


def _log_train_info(logger, algo, epoch, total, metrics, em, avg):
    first = _fmt_metrics(metrics)[:3]
    logger.info(
        "[%s] Epoch %03d/%03d | eval=%.2f±%.2f win=%.0f%% len=%.1f | reward20=%.2f | %s",
        algo.upper(), epoch, total,
        em["mean_reward"], em["std_reward"],
        em["win_rate"] * 100, em["mean_length"], avg,
        " ".join(first),
    )


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="JSON 配置路径，默认 fz_agent/configs/ppo_combat.json")
    args = parser.parse_args()

    default_paths = [
        _THIS_DIR / "configs" / "ppo_combat.json",
        _PROJECT_ROOT / "docs" / "1.json",
    ]
    config_path = args.config
    if config_path is None:
        for p in default_paths:
            if p.exists():
                config_path = p
                break

    if config_path is None or not Path(config_path).exists():
        print("未找到默认 JSON 配置，请指定 --config")
        return

    cfg = load_json_config(str(config_path))
    print(f"Loaded config: {config_path}")

    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    seed = train_cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "cuda"))

    env = make_env(env_cfg, seed=seed)
    eval_env = make_env(env_cfg, seed=seed + 1000)

    agent, roles = build_rl_from_json_config(cfg)
    algo_name = cfg["algo"]["name"].lower()
    run_name = cfg.get("meta", {}).get("name", f"{algo_name}_combat")

    logger = TrainLogger(run_name=run_name, use_tensorboard=True, plot=True)
    logger.info("Device: %s  Algo: %s  roles=%s", device, algo_name, list(roles.keys()))
    logger.log_config(cfg)

    discrete_on_policy = {"ppo", "a2c"}
    mc = {"reinforce"}
    discrete_off_policy = {"dqn", "ddqn", "dueling_dqn", "rainbow"}
    continuous = {"ddpg", "td3", "sac"}

    if algo_name in discrete_on_policy:
        train_discrete_ppo_like(env, eval_env, agent, cfg, logger, algo_name)
    elif algo_name in mc:
        train_reinforce(env, eval_env, agent, cfg, logger)
    elif algo_name in discrete_off_policy:
        train_dqn_like(env, eval_env, agent, cfg, logger, algo_name)
    elif algo_name in continuous:
        train_continuous(env, eval_env, agent, cfg, logger, algo_name)
    else:
        logger.info("算法 %s 无训练循环，仅测试 act()", algo_name)
        obs, _ = env.reset()
        a = agent.act(obs, deterministic=True)
        logger.info("act() OK: action=%s", a)
        logger.close()


if __name__ == "__main__":
    main()

"""通用入口：JSON 配置驱动的训练与推理。

子命令：
    rl-train   强化学习训练（并导出构建体）
    rl-exp     强化学习推理（加载构建体在环境中运行）
    dl-train   深度学习训练（网络 + 数据集 → 权重）
    dl-exp     深度学习推理
    serve      启动多构建体推理服务

示例：
    python -m fz_agent.main rl-train --config fz_agent/configs/ppo_combat.json
    python -m fz_agent.main rl-exp   --bundle fz_agent/bundles/ppo_combat --episodes 3
    python -m fz_agent.main dl-train --config fz_agent/configs/dl_regression.json
    python -m fz_agent.main serve    --bundle-dir fz_agent/bundles --port 8000
"""
import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fz_agent.env.CombatEnv import CombatEnv, CombatEnvProxy
from fz_agent.env.base import IOSpec
from fz_agent.dl.factory import (
    build_rl_from_json_config,
    build_backbone_from_stack,
    build_backbone_from_graph,
    load_json_config,
)
from fz_agent.rl.dqn import ReplayBuffer
from fz_agent.utils.logger import TrainLogger


# ============================================================
# 通用工具
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


def _default_config_path() -> str:
    for p in (_THIS_DIR / "configs" / "ppo_combat.json", _PROJECT_ROOT / "docs" / "1.json"):
        if p.exists():
            return str(p)
    return ""


def _load_config(config_path=None) -> dict:
    path = config_path or _default_config_path()
    if not path or not Path(path).exists():
        raise FileNotFoundError("未找到 JSON 配置，请通过 --config 指定")
    cfg = load_json_config(path)
    print(f"Loaded config: {path}")
    return cfg


def make_env(env_cfg: dict, seed: int = 42):
    """按配置构建环境（真实引擎在此替换）。"""
    env_type = env_cfg.get("type", "combat")
    io_cfg = env_cfg.get("io", {}) or {}
    action_cfg = io_cfg.get("action", {}) or {}
    space_type = env_cfg.get("action_space_type", action_cfg.get("space_type", "discrete"))
    action_dim = env_cfg.get("action_dim", action_cfg.get("dim", 3))

    if env_type not in ("combat", "combat_proxy"):
        raise ValueError(f"未知 env type: {env_type}（当前内置 combat，真实引擎请实现 BaseEnv）")

    params = dict(env_cfg.get("config", {}))
    params.setdefault("num_enemies", 3)
    params.setdefault("max_detected", 10)
    params.setdefault("max_episode_steps", 300)
    params["seed"] = seed
    params["action_space_type"] = space_type
    params["action_dim"] = action_dim

    env = CombatEnvProxy(**params)
    if io_cfg:
        env.io_spec = IOSpec.from_dict(io_cfg)
    return env


def _to_env_action(action, env):
    """把算法输出统一成环境可消费的动作：离散→int，连续→一维 ndarray。"""
    discrete = getattr(env.io_spec.action, "space_type", "discrete") == "discrete"
    if isinstance(action, torch.Tensor):
        a = action.detach().cpu().flatten()
        return int(a[0]) if discrete else a.numpy()
    if isinstance(action, np.ndarray):
        a = action.reshape(-1)
        return int(a[0]) if discrete else a
    return int(action) if discrete else action


@torch.no_grad()
def evaluate(env, agent, episodes=5):
    rewards, lengths = [], []
    wins = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        total, steps, result = 0.0, 0, None
        while True:
            action = _to_env_action(agent.act(obs, deterministic=True), env)
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
    for attr in ("target", "critic", "target_critic", "q1", "q2", "target_q1", "target_q2"):
        net = getattr(agent, attr, None)
        if net is not None:
            state[attr] = net.state_dict()
    for attr in ("optimizer", "actor_optimizer", "critic_optimizer", "q1_optimizer", "q2_optimizer"):
        opt = getattr(agent, attr, None)
        if opt is not None:
            state[attr] = opt.state_dict()
    torch.save(state, path)


def export_bundle(agent, cfg: dict, env, run_name: str) -> str:
    """训练结束导出构建体：io + 网络结构 + 权重。"""
    from fz_agent.construct import AgentBundle

    bundle = AgentBundle.from_agent(
        agent, cfg, env.io_spec, bundle_id=run_name,
        metadata={
            "algorithm": cfg.get("algo", {}).get("name", ""),
            "env_type": cfg.get("env", {}).get("type", "combat"),
        },
    )
    out = bundle.save(str(_THIS_DIR / "bundles" / run_name))
    print(f"[bundle] 构建体已导出: {out}")
    return out


# ============================================================
# 采样函数
# ============================================================

def collect_rollout_on_policy(env, agent, obs, rollout_steps=1024, env_steps_start=0):
    buffer = {
        "obs": [], "actions": [], "log_probs": [],
        "rewards": [], "values": [], "next_values": [],
        "terminated": [], "episode_ends": [],
    }
    ep_records = []
    cur_rew, cur_steps = 0.0, 0

    for i in range(rollout_steps):
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
            traj["log_probs"].append(lp)

            obs, r, terminated, truncated, _ = env.step(int(a))
            traj["rewards"].append(float(r))
            done = terminated or truncated
        trajectories.append(traj)
    return trajectories


def _sample_tensor_batch(rb: ReplayBuffer, batch_size: int, device):
    obs, actions, rewards, next_obs, dones = rb.sample(batch_size)
    return {
        "obs": torch.as_tensor(obs, dtype=torch.float32, device=device),
        "action": torch.as_tensor(actions, dtype=torch.float32, device=device),
        "reward": torch.as_tensor(rewards, dtype=torch.float32, device=device),
        "next_obs": torch.as_tensor(next_obs, dtype=torch.float32, device=device),
        "done": torch.as_tensor(dones, dtype=torch.float32, device=device),
    }


# ============================================================
# 训练循环
# ============================================================

def train_discrete_ppo_like(env, eval_env, agent, cfg, logger, algo_name):
    algo_cfg = cfg["algo"]["config"]
    train_cfg = cfg.get("train", {})

    total_epochs = train_cfg.get("total_epochs", 50)
    rollout_steps = train_cfg.get("rollout_steps", 1024)
    update_epochs = train_cfg.get("update_epochs", algo_cfg.get("ppo_epochs", 4))
    minibatch_size = train_cfg.get("minibatch_size", algo_cfg.get("batch_size", 128))
    eval_interval = train_cfg.get("eval_interval", 5)

    recent = deque(maxlen=20)
    obs, _ = env.reset()
    env_steps = 0

    for epoch in range(1, total_epochs + 1):
        rollout, obs, ep_records = collect_rollout_on_policy(
            env, agent, obs, rollout_steps=rollout_steps, env_steps_start=env_steps)
        env_steps += rollout_steps
        for r in ep_records:
            logger.log_episode(**{k: r[k] for k in ("reward", "length", "result", "env_steps")})
            recent.append(r["reward"])

        if algo_name == "ppo":
            metrics = agent.update(rollout, update_epochs=update_epochs,
                                   minibatch_size=minibatch_size)
        elif algo_name == "a2c":
            batch = {k: torch.tensor(v, dtype=torch.float32, device=agent.device)
                     for k, v in rollout.items() if k != "obs"}
            batch["log_probs"] = batch["log_probs"].unsqueeze(-1)
            metrics = agent.update(batch) or {"loss": 0.0}
        else:
            metrics = {"loss": 0.0}

        avg = float(np.mean(recent)) if recent else float("nan")
        logger.log_epoch(epoch=epoch, metrics=metrics,
                         extra={"env_steps": env_steps, "train_reward_mean20": avg})

        if epoch % eval_interval == 0 or epoch == 1:
            em = evaluate(eval_env, agent, episodes=5)
            logger.log_eval(epoch, em)
            _log_train_info(logger, algo_name, epoch, total_epochs, metrics, em, avg)
        else:
            logger.info("[%s] Epoch %03d | reward20=%.2f | %s", algo_name.upper(), epoch, avg,
                        " ".join(_fmt_metrics(metrics))[:60])

        if epoch % max(1, total_epochs // 5) == 0:
            save_checkpoint(agent, epoch,
                            str(_THIS_DIR / "checkpoints" / f"{algo_name}_epoch_{epoch}.pth"))

    save_checkpoint(agent, total_epochs, str(_THIS_DIR / "checkpoints" / f"{algo_name}_final.pth"))
    logger.close(extra_summary={"algo": algo_name})


def train_reinforce(env, eval_env, agent, cfg, logger):
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
            lp_tensor = (torch.stack(lp_list).to(agent.device) if isinstance(lp_list[0], torch.Tensor)
                         else torch.as_tensor(lp_list, dtype=torch.float32, device=agent.device))
            batch = {"log_probs": lp_tensor, "rewards": t["rewards"]}
            try:
                m = agent.update(batch) or {"loss": 0.0}
                for k, v in m.items():
                    all_metrics.setdefault(k, []).append(v)
            except Exception as e:  # noqa: BLE001
                all_metrics.setdefault("error", []).append(str(e)[:80])

            R = sum(t["rewards"])
            recent.append(R)
            logger.log_episode(reward=R, length=len(t["rewards"]), result="rollout",
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
            logger.info("[REINFORCE] Epoch %03d | reward20=%.2f | %s", epoch, avg,
                        " ".join(_fmt_metrics(metrics))[:60])

    save_checkpoint(agent, total_epochs, str(_THIS_DIR / "checkpoints" / "reinforce_final.pth"))
    logger.close(extra_summary={"algo": "reinforce"})


def train_dqn_like(env, eval_env, agent, cfg, logger, algo_name):
    algo_cfg = cfg["algo"]["config"]
    train_cfg = cfg.get("train", {})

    total_steps = train_cfg.get("total_steps", 20000)
    eval_interval = train_cfg.get("eval_interval_steps", 2000)
    rb_capacity = train_cfg.get("capacity", algo_cfg.get("capacity", 10000))
    warmup_steps = train_cfg.get("warmup_steps", 1000)

    rb = ReplayBuffer(capacity=rb_capacity)
    recent = deque(maxlen=20)
    obs, _ = env.reset()
    cur_rew, cur_steps = 0.0, 0

    for step in range(1, total_steps + 1):
        explore = step <= warmup_steps or (
            hasattr(agent, "epsilon") and random.random() < agent.epsilon)
        action = agent.act(obs, deterministic=not explore)
        if isinstance(action, torch.Tensor):
            action = int(action.item()) if action.dim() == 0 else int(action.squeeze())

        next_obs, reward, terminated, truncated, _ = env.step(action)
        rb.push(obs, action, float(reward), next_obs, terminated or truncated)

        cur_rew += reward
        cur_steps += 1
        obs = next_obs

        try:
            agent.update(rb)
        except Exception:  # noqa: BLE001
            pass

        if terminated or truncated:
            recent.append(cur_rew)
            logger.log_episode(reward=cur_rew, length=cur_steps, result="ended", env_steps=step)
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
    """DDPG / TD3 / SAC：连续动作 + 经验回放。

    环境通过 action_space_type=continuous 把连续动作映射到战斗语义
    （见 CombatEnv._resolve_action），因此同一套训练循环可复用。
    """
    algo_cfg = cfg["algo"]["config"]
    train_cfg = cfg.get("train", {})

    total_steps = train_cfg.get("total_steps", 20000)
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    eval_interval = train_cfg.get("eval_interval_steps", 2000)
    rb_capacity = train_cfg.get("capacity", algo_cfg.get("capacity", 50000))
    batch_size = algo_cfg.get("batch_size", 64)
    act_dim = int(algo_cfg.get("action_dim", 3))

    rb = ReplayBuffer(capacity=rb_capacity)
    recent = deque(maxlen=20)
    obs, _ = env.reset()
    cur_rew, cur_steps = 0.0, 0

    for step in range(1, total_steps + 1):
        if step <= warmup_steps:
            action = np.random.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)
        else:
            raw = agent.act(obs, deterministic=False)
            action = np.asarray(raw, dtype=np.float32).reshape(-1)
            if action.size < act_dim:  # 兜底：异常输出直接随机
                action = np.random.uniform(-1.0, 1.0, size=(act_dim,)).astype(np.float32)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        rb.push(obs, np.asarray(action, dtype=np.float32), float(reward),
                next_obs, terminated or truncated)

        cur_rew += reward
        cur_steps += 1
        obs = next_obs

        if step > warmup_steps and len(rb) >= batch_size:
            try:
                if algo_name == "sac":
                    metrics = agent.update(rb.sample(batch_size))
                else:
                    metrics = agent.update(
                        _sample_tensor_batch(rb, batch_size, agent.device))
                if metrics and step % eval_interval == 0:
                    logger.info("[%s] step %05d | %s", algo_name.upper(), step,
                                " ".join(_fmt_metrics(metrics))[:80])
            except Exception:  # noqa: BLE001
                pass

        if terminated or truncated:
            recent.append(cur_rew)
            logger.log_episode(reward=cur_rew, length=cur_steps, result="ended", env_steps=step)
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
    logger.info(
        "[%s] Epoch %03d/%03d | eval=%.2f±%.2f win=%.0f%% len=%.1f | reward20=%.2f | %s",
        algo.upper(), epoch, total,
        em["mean_reward"], em["std_reward"], em["win_rate"] * 100,
        em["mean_length"], avg, " ".join(_fmt_metrics(metrics)[:3]),
    )


# ============================================================
# 强化学习：训练 / 推理
# ============================================================

def mainRL_train(config_path=None):
    cfg = _load_config(config_path)
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
    logger.info("Device: %s  Algo: %s  roles=%s  io=%s",
                device, algo_name, list(roles.keys()), env.io_spec.to_dict())
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
        logger.info("act() OK: action=%s", agent.act(obs, deterministic=True))
        logger.close()

    export_bundle(agent, cfg, env, run_name)
    return agent


def mainRL_exp(bundle_path=None, config_path=None, weights_path=None,
               episodes=3, seed=42):
    """强化学习推理：加载构建体（或 config+weights），在环境中跑若干回合。"""
    from fz_agent.construct import AgentBundle, ModelPool

    if bundle_path:
        bundle = AgentBundle.load(bundle_path)
        agent = bundle.build_agent()
        io_spec = IOSpec.from_dict(bundle.io_spec)
        cfg = bundle.config
        print(f"[exp] 加载构建体 {bundle_path} (id={bundle.bundle_id})")
    else:
        cfg = _load_config(config_path)
        agent, _ = build_rl_from_json_config(cfg)
        io_spec = IOSpec.from_dict(cfg.get("io", {}))
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            agent.policy.load_state_dict(state.get("policy", state))
            print(f"[exp] 加载权重 {weights_path}")
        agent.policy.eval()

    env_cfg = cfg.get("env", {})
    env = make_env(env_cfg, seed=seed)
    env.io_spec = io_spec

    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, ret, steps = False, 0.0, 0
        while not done:
            action = _to_env_action(agent.act(obs, deterministic=True), env)
            print(f"  ep{ep} step{steps:03d} obs_shape={np.asarray(obs).shape} action={action}")
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            steps += 1
            done = terminated or truncated
        total_rewards.append(ret)
        print(f"[exp] episode {ep}: return={ret:.2f} steps={steps} "
              f"result={info.get('result', 'n/a') if isinstance(info, dict) else 'n/a'}")
    print(f"[exp] 平均回报: {np.mean(total_rewards):.2f}")
    return total_rewards


# ============================================================
# 深度学习：训练 / 推理
# ============================================================

class _DLModel(nn.Module):
    """backbone + 线性输出头（监督学习用）。"""

    def __init__(self, backbone, feat_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, out_dim)

    def _pool(self, x):
        out = self.backbone(x)
        if out.dim() == 4:
            return out.flatten(1)
        if out.dim() == 3:
            return out.mean(1)  # 实体/序列维度平均，保留 batch
        return out

    def forward(self, x):
        return self.head(self._pool(x))


def _build_dl_backbone(cfg: dict):
    graph = cfg.get("network_graph")
    if graph:
        return build_backbone_from_graph(graph), True
    stack = cfg.get("network_stack")
    if stack is None and "networks" in cfg:
        first = next(iter(cfg["networks"].values()))
        stack = first.get("network_stack") if isinstance(first, dict) else first
    return build_backbone_from_stack(stack or []), False


def _load_npz(path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    data = np.load(path)
    x = data["X"] if "X" in data else data[data.files[0]]
    y = data["Y"] if "Y" in data else data[data.files[-1]]
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def mainDL_train(config_path=None):
    """深度学习训练：JSON 定义的网络 + 数据集 → 权重文件。"""
    cfg = _load_config(config_path)
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    name = cfg.get("meta", {}).get("name", "dl_model")

    x, y = _load_npz(data_cfg.get("path", ""))
    task = data_cfg.get("task", "regression").lower()
    if task == "classification":
        y = y.reshape(-1).astype(np.int64)
        out_dim = int(y.max()) + 1
        y_tensor = torch.as_tensor(y, dtype=torch.long)
    else:
        y = y.reshape(len(y), -1)
        out_dim = y.shape[1]
        y_tensor = torch.as_tensor(y, dtype=torch.float32)

    backbone, _ = _build_dl_backbone(cfg)
    device = get_device(cfg.get("device", "cuda"))
    backbone = backbone.to(device)

    with torch.no_grad():
        sample = torch.as_tensor(x[:2], dtype=torch.float32, device=device)
        out = backbone(sample)
        if out.dim() == 4:
            feat_dim = int(np.prod(out.shape[1:]))
        elif out.dim() == 3:
            feat_dim = int(out.shape[-1])
        else:
            feat_dim = int(out.shape[1])
    model = _DLModel(backbone, feat_dim, out_dim).to(device)

    batch_size = int(data_cfg.get("batch_size", train_cfg.get("batch_size", 64)))
    epochs = int(train_cfg.get("epochs", data_cfg.get("epochs", 20)))
    lr = float(train_cfg.get("lr", data_cfg.get("lr", 1e-3)))
    val_split = float(data_cfg.get("val_split", 0.1))

    n = len(x)
    n_val = int(n * val_split)
    perm = np.random.default_rng(train_cfg.get("seed", 42)).permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_t = torch.as_tensor(x, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    logger = TrainLogger(run_name=name, use_tensorboard=True, plot=True)
    logger.log_config(cfg)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = train_idx.copy()
        np.random.shuffle(idx)
        losses = []
        for start in range(0, len(idx), batch_size):
            mb = idx[start:start + batch_size]
            xb = x_t[mb].to(device)
            yb = y_tensor[mb].to(device)
            pred = model(xb)
            if task == "classification":
                loss = loss_fn(pred, yb)
            else:
                loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_loss = _evaluate_dl(model, x_t, y_tensor, val_idx, device, loss_fn, batch_size)
        logger.log_epoch(epoch=epoch, metrics={"loss": train_loss, "val_loss": val_loss})
        logger.info("[DL] Epoch %03d/%03d | loss=%.6f val_loss=%.6f",
                    epoch, epochs, train_loss, val_loss)

    out_dir = _THIS_DIR / "output" / "dl_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.pt"
    torch.save({"config": cfg, "io": cfg.get("io", {}),
                "state_dict": model.state_dict()}, out_path)
    print(f"[DL] 模型已保存: {out_path}")
    logger.close(extra_summary={"task": task, "val_loss": val_loss})
    return str(out_path)


def _evaluate_dl(model, x_t, y_tensor, idx, device, loss_fn, batch_size):
    if len(idx) == 0:
        return 0.0
    model.eval()
    losses = []
    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            mb = idx[start:start + batch_size]
            pred = model(x_t[mb].to(device))
            losses.append(loss_fn(pred, y_tensor[mb].to(device)).item())
    return float(np.mean(losses)) if losses else 0.0


def mainDL_exp(model_path=None, config_path=None, input_path=None, limit=5):
    """深度学习推理：加载权重，对输入数据或单条样本给出预测。"""
    if model_path is None:
        # 从配置推断默认模型路径
        cfg = _load_config(config_path)
        name = cfg.get("meta", {}).get("name", "dl_model")
        model_path = str(_THIS_DIR / "output" / "dl_models" / f"{name}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    device = get_device(cfg.get("device", "cuda"))
    backbone, _ = _build_dl_backbone(cfg)
    state = ckpt["state_dict"]
    head_w = state.get("head.weight")
    feat_dim = head_w.shape[1] if head_w is not None else 1
    out_dim = head_w.shape[0] if head_w is not None else 1
    model = _DLModel(backbone, feat_dim, out_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    if input_path:
        x, _ = _load_npz(input_path)
    else:
        io = ckpt.get("io", {})
        dim = int(io.get("obs", {}).get("shape", [3])[-1]) if io else 3
        x = np.random.randn(1, dim).astype(np.float32)
        print(f"[DL] 未提供输入文件，使用随机样本 shape={x.shape}")

    with torch.no_grad():
        pred = model(torch.as_tensor(x[:limit], dtype=torch.float32, device=device))
    print(f"[DL] 预测输出 shape={tuple(pred.shape)}:")
    print(pred.cpu().numpy())
    return pred.cpu().numpy()


# ============================================================
# CLI
# ============================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="FZ agent 训练/推理入口")
    sub = parser.add_subparsers(dest="command")

    p_rl_train = sub.add_parser("rl-train", help="强化学习训练")
    p_rl_train.add_argument("--config", "-c", default=None)

    p_rl_exp = sub.add_parser("rl-exp", help="强化学习推理")
    p_rl_exp.add_argument("--bundle", default=None, help="构建体目录")
    p_rl_exp.add_argument("--config", "-c", default=None)
    p_rl_exp.add_argument("--weights", default=None)
    p_rl_exp.add_argument("--episodes", type=int, default=3)
    p_rl_exp.add_argument("--seed", type=int, default=42)

    p_dl_train = sub.add_parser("dl-train", help="深度学习训练")
    p_dl_train.add_argument("--config", "-c", default=None)

    p_dl_exp = sub.add_parser("dl-exp", help="深度学习推理")
    p_dl_exp.add_argument("--model", default=None)
    p_dl_exp.add_argument("--config", "-c", default=None)
    p_dl_exp.add_argument("--input", default=None)
    p_dl_exp.add_argument("--limit", type=int, default=5)

    p_serve = sub.add_parser("serve", help="启动多构建体推理服务")
    p_serve.add_argument("--bundle-dir", default=None)
    p_serve.add_argument("--bundle", action="append", default=None)
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    if args.command is None:
        # 兼容旧用法：无子命令时默认跑强化学习训练
        return mainRL_train(None)
    if args.command == "rl-train":
        return mainRL_train(args.config)
    if args.command == "rl-exp":
        return mainRL_exp(args.bundle, args.config, args.weights, args.episodes, args.seed)
    if args.command == "dl-train":
        return mainDL_train(args.config)
    if args.command == "dl-exp":
        return mainDL_exp(args.model, args.config, args.input, args.limit)
    if args.command == "serve":
        from fz_agent.serve import run_server
        return run_server(args.bundle_dir, args.bundle, args.host, args.port)
    parser.print_help()
    return None


if __name__ == "__main__":
    main()

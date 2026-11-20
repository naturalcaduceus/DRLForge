# -*- coding: utf-8 -*-
"""智能体构建/训练平台的纯逻辑层（不依赖 tkinter，便于单元测试）。

gui_agent_studio.py 只做界面绑定，所有"把表单变成 fz_agent 能解释的 JSON"、
"校验"、"解析训练输出"、"读曲线"的逻辑都在这里，因此可以脱离界面测试。

对齐关系（甲方要求 → 代码）：
    11 种算法      ALGOS（9 种可直接训练，Rainbow/IQN 后端待补齐，UI 会标注）
    8 种网络积木   NET_MODULES（mlp/cnn/conv1d/lstm/gru/attention/transformer/resnet）
    6 种演化训练   STRATEGIES（transfer/adversarial/incremental/distill/multitask/ensemble）
    配置驱动        build_config() 产出完整 cfg（meta/env/io/networks/algo/strategy/train）
"""
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FZ_ROOT = PROJECT_ROOT / "fz_agent"
CONFIG_DIR = FZ_ROOT / "configs"
BUNDLE_DIR = FZ_ROOT / "bundles"
RUNS_DIR = FZ_ROOT / "runs"
DEFAULT_REPLAY_DIR = PROJECT_ROOT.parent / "chess-engine2" / "logs" / "runs"

# ---------------- 算法库（显示名 → 后端名 / 算法族 / 角色） ----------------
# roles 与 fz_agent/dl/factory.py 的 ALGO_REQUIREMENTS 保持一致（这里静态保存，
# 避免 GUI 启动时就去 import torch）。
ALGOS = [
    {"label": "PPO", "name": "ppo", "family": "on_policy",
     "roles": {"policy": "actor_critic"}, "ok": True},
    {"label": "A2C", "name": "a2c", "family": "on_policy",
     "roles": {"policy": "actor_critic"}, "ok": True},
    {"label": "REINFORCE", "name": "reinforce", "family": "on_policy",
     "roles": {"policy": "actor_discrete"}, "ok": True},
    {"label": "DQN", "name": "dqn", "family": "off_policy",
     "roles": {"q_net": "q"}, "ok": True},
    {"label": "DDQN", "name": "ddqn", "family": "off_policy",
     "roles": {"q_net": "q"}, "ok": True},
    {"label": "Dueling DQN", "name": "dueling_dqn", "family": "off_policy",
     "roles": {"q_net": "dueling_q"}, "ok": True},
    {"label": "DDPG", "name": "ddpg", "family": "continuous",
     "roles": {"actor": "actor_continuous", "critic": "critic_q_continuous"}, "ok": True},
    {"label": "TD3", "name": "td3", "family": "continuous",
     "roles": {"actor": "actor_continuous", "q1": "critic_q_continuous",
               "q2": "critic_q_continuous"}, "ok": True},
    {"label": "SAC", "name": "sac", "family": "continuous",
     "roles": {"actor": "actor_sac", "q1": "critic_q_continuous",
               "q2": "critic_q_continuous"}, "ok": True},
    # 这两种在 rl/ 里有类，但构造签名尚未适配 BaseRLAlgo/JSON 工厂：
    # 先保留在算法库中（甲方要求 ≥11 种），UI 标注"后端待补齐"，校验会拦截。
    {"label": "Rainbow（后端待补齐）", "name": "rainbow", "family": "off_policy",
     "roles": {"q_net": "q"}, "ok": False},
    {"label": "IQN（后端待补齐）", "name": "iqn", "family": "off_policy",
     "roles": {"q_net": "q"}, "ok": False},
]
ALGO_BY_NAME = {a["name"]: a for a in ALGOS}

# ---------------- 网络积木（8 种，参数定义与 blocks/*.py 对齐） ----------------
NET_MODULES = [
    {"label": "MLP", "key": "mlp", "params": [
        ("in_dim", int, 10), ("hidden_dims", str, "128,128"),
        ("out_dim", int, 128), ("activation", str, "ReLU")]},
    {"label": "CNN", "key": "cnn", "params": [
        ("in_channels", int, 3), ("channels", str, "32,64"),
        ("kernel_sizes", str, "3,3"), ("strides", str, "1,1"),
        ("padding", int, 0), ("activation", str, "ReLU")]},
    {"label": "Conv1d", "key": "conv1d", "params": [
        ("in_channels", int, 8), ("channels", str, "32,32"),
        ("kernel_sizes", str, "3,3"), ("strides", str, "1,1"),
        ("activation", str, "ReLU")]},
    {"label": "LSTM", "key": "lstm", "params": [
        ("input_size", int, 10), ("hidden_size", int, 64), ("num_layers", int, 1),
        ("out_dim", int, 64), ("batch_first", str, "True"),
        ("bidirectional", str, "False")]},
    {"label": "GRU", "key": "gru", "params": [
        ("input_size", int, 10), ("hidden_size", int, 64), ("num_layers", int, 1),
        ("out_dim", int, 64), ("batch_first", str, "True"),
        ("bidirectional", str, "False")]},
    {"label": "SelfAttention", "key": "attention", "params": [
        ("embed_dim", int, 64), ("num_heads", int, 4)]},
    {"label": "Transformer", "key": "transformer", "params": [
        ("d_model", int, 64), ("nhead", int, 4), ("dim_feedforward", int, 128),
        ("num_layers", int, 2), ("dropout", float, 0.1)]},
    {"label": "ResNet", "key": "resnet", "params": [
        ("in_channels", int, 32), ("out_channels", int, 64), ("stride", int, 1)]},
]
NET_BY_KEY = {m["key"]: m for m in NET_MODULES}

# ---------------- 演化训练策略（6 种，与 stra/strategy_wrapper.py 对齐） ----------------
STRATEGIES = [
    {"name": "", "label": "（无）"},
    {"name": "transfer_learning", "label": "迁移学习",
     "fields": [("pretrained", str, ""), ("freeze", str, "False")]},
    {"name": "adversarial_train", "label": "对抗训练",
     "fields": [("epsilon", float, 0.01)]},
    {"name": "incremental_continual", "label": "增量持续训练", "fields": []},
    {"name": "distill_light", "label": "模型蒸馏轻量化",
     "fields": [("temperature", float, 2.0), ("alpha", float, 0.7)]},
    {"name": "multitask_joint", "label": "多任务联合", "fields": []},
    {"name": "ensemble_vote", "label": "集成投票", "fields": []},
]
STRATEGY_BY_NAME = {s["name"]: s for s in STRATEGIES}

# ---------------- 预设（保证第一次打开就能生成可训练的配置） ----------------
PRESETS = {
    "chess_ppo": {
        "name": "ppo_chess_gui", "desc": "chess-engine2 红方 PPO",
        "env": {"type": "chess_remote", "url": "", "side": "red",
                "max_entities": 12, "max_actions": 32},
        "io": {"obs_dim": 10, "entity_set": True, "space_type": "discrete",
               "action_dim": 32,
               "names": ["dx", "dy", "hp_ratio", "atk_ratio", "armor",
                         "move_ratio", "is_enemy", "is_self", "alive", "tick_ratio"]},
        "algo": "ppo", "stack": [{"type": "mlp", "config": {
            "in_dim": 10, "hidden_dims": [128, 128], "out_dim": 128,
            "activation": "ReLU"}}],
        "train": {"seed": 42, "total_epochs": 10, "rollout_steps": 512,
                  "update_epochs": 4, "minibatch_size": 128, "eval_interval": 5},
    },
    "chess_dqn": {
        "name": "dqn_chess_gui", "desc": "chess-engine2 红方 DQN",
        "env": {"type": "chess_remote", "url": "", "side": "red",
                "max_entities": 12, "max_actions": 32},
        "io": {"obs_dim": 10, "entity_set": True, "space_type": "discrete",
               "action_dim": 32,
               "names": ["dx", "dy", "hp_ratio", "atk_ratio", "armor",
                         "move_ratio", "is_enemy", "is_self", "alive", "tick_ratio"]},
        "algo": "dqn", "stack": [{"type": "mlp", "config": {
            "in_dim": 10, "hidden_dims": [128, 128], "out_dim": 128,
            "activation": "ReLU"}}],
        "train": {"seed": 42, "total_steps": 5000, "capacity": 20000,
                  "warmup_steps": 500, "eval_interval_steps": 1000},
    },
    "combat_ppo": {
        "name": "ppo_combat_gui", "desc": "内置 CombatEnv PPO（无需环境服务）",
        "env": {"type": "combat", "url": "", "side": "red",
                "num_enemies": 3, "max_detected": 10, "max_episode_steps": 300},
        "io": {"obs_dim": 3, "entity_set": True, "space_type": "discrete",
               "action_dim": 3, "names": ["dx", "dy", "hp_ratio"]},
        "algo": "ppo", "stack": [{"type": "mlp", "config": {
            "in_dim": 3, "hidden_dims": [128, 128], "out_dim": 128,
            "activation": "ReLU"}}],
        "train": {"seed": 42, "total_epochs": 20, "rollout_steps": 512,
                  "update_epochs": 4, "minibatch_size": 128, "eval_interval": 5},
    },
}

# 各算法默认超参（与 gui_json_builder / configs 对齐）
ALGO_DEFAULTS = {
    "ppo": {"learning_rate": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_coef": 0.2, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5},
    "a2c": {"learning_rate": 3e-4, "gamma": 0.99},
    "reinforce": {"learning_rate": 1e-3, "gamma": 0.98},
    "dqn": {"learning_rate": 1e-3, "gamma": 0.99, "tau": 0.005,
            "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.995,
            "batch_size": 64},
    "ddqn": {"learning_rate": 1e-3, "gamma": 0.99, "tau": 0.005,
             "epsilon_start": 1.0, "epsilon_end": 0.05, "epsilon_decay": 0.995,
             "batch_size": 64},
    "dueling_dqn": {"learning_rate": 1e-3, "gamma": 0.99, "tau": 0.005,
                    "epsilon_start": 1.0, "epsilon_end": 0.05,
                    "epsilon_decay": 0.995, "batch_size": 64},
    "ddpg": {"learning_rate": 1e-3, "gamma": 0.99, "tau": 0.005, "batch_size": 64},
    "td3": {"learning_rate": 1e-3, "gamma": 0.99, "tau": 0.005, "batch_size": 64},
    "sac": {"learning_rate": 3e-4, "gamma": 0.99, "alpha": 0.2, "batch_size": 64},
    "rainbow": {"learning_rate": 1e-4, "gamma": 0.99},
    "iqn": {"learning_rate": 1e-4, "gamma": 0.99},
}


def parse_value(raw, typ):
    """表单字符串 → 目标类型；str 支持逗号列表与 True/False。"""
    s = str(raw).strip()
    if typ is int:
        return int(s)
    if typ is float:
        return float(s)
    if typ is bool:
        return s.lower() in ("true", "1", "yes", "on")
    if typ is str:
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        if "," in s:
            out = []
            for part in s.split(","):
                part = part.strip()
                try:
                    out.append(int(part))
                except ValueError:
                    try:
                        out.append(float(part))
                    except ValueError:
                        out.append(part)
            return out
        return s
    return s


def default_algo_config(algo_name, action_dim, space_type):
    cfg = dict(ALGO_DEFAULTS.get(algo_name, {}))
    cfg["action_dim"] = int(action_dim)
    cfg["device"] = "cpu"
    if space_type == "continuous":
        cfg.setdefault("batch_size", 64)
    return cfg


def _normalize_stack(stack, obs_dim, warnings):
    """把网络堆叠里与观测维度相关的入口参数对齐到 obs_dim。"""
    out = []
    for layer in stack:
        layer = copy.deepcopy(layer)
        key = layer.get("type", "").lower()
        cfg = layer.setdefault("config", {})
        if key == "mlp":
            if int(cfg.get("in_dim", obs_dim)) != int(obs_dim):
                warnings.append("MLP.in_dim 已从 %s 对齐到观测维度 %s"
                                % (cfg.get("in_dim"), obs_dim))
            cfg["in_dim"] = int(obs_dim)
        elif key in ("lstm", "gru"):
            if int(cfg.get("input_size", obs_dim)) != int(obs_dim):
                warnings.append("%s.input_size 已对齐到观测维度 %s" % (key, obs_dim))
            cfg["input_size"] = int(obs_dim)
        elif key in ("cnn", "resnet", "conv1d"):
            warnings.append("%s 需要图像/序列输入，与实体集合观测 [N, %d] 不兼容，"
                            "请确认后再训练" % (key, obs_dim))
        out.append(layer)
    return out


def align_stack(stack, obs_dim):
    """公共包装：把堆叠入口维度对齐到 obs_dim，返回 (stack, warnings)。"""
    warnings = []
    return _normalize_stack(stack, int(obs_dim), warnings), warnings


def build_config(form):
    """表单 dict → fz_agent 完整配置（可直接交给 main.py rl-train/dl-train）。

    form 结构见 PRESETS；缺失字段用默认值补齐。
    """
    paradigm = form.get("paradigm", "rl")
    name = (form.get("name") or "agent").strip().replace(" ", "_")
    if paradigm == "dl":
        return _build_dl_config(form, name)

    warnings = []
    io = dict(form.get("io") or {})
    env = dict(form.get("env") or {})
    obs_dim = int(io.get("obs_dim", 10))
    entity_set = bool(io.get("entity_set", True))
    space_type = io.get("space_type", "discrete")
    action_dim = int(io.get("action_dim", 32))
    max_entities = int(env.get("max_entities", form.get("max_entities", 12)))

    algo_name = form.get("algo", "ppo")
    algo_meta = ALGO_BY_NAME.get(algo_name, ALGO_BY_NAME["ppo"])
    stack = _normalize_stack(form.get("stack") or [], obs_dim, warnings)

    networks = {}
    for role, head in algo_meta["roles"].items():
        networks[role] = {"network_stack": copy.deepcopy(stack), "head": head}

    algo_cfg = form.get("algo_config") or default_algo_config(
        algo_name, action_dim, space_type)
    algo_cfg = dict(algo_cfg)
    algo_cfg["action_dim"] = action_dim

    cfg = {
        "meta": {"name": name, "desc": form.get("desc", "")},
        "env": {
            "type": env.get("type", "chess_remote"),
            "url": env.get("url") or "",
            "config": {
                "side": env.get("side", "red"),
                "max_entities": max_entities,
                "timeout": env.get("timeout", 15.0),
            },
            "port": int(env.get("port", 8000)),
            "action_space_type": space_type,
            "action_dim": action_dim,
            "io": {
                "obs": {"shape": [max_entities, obs_dim], "dim": obs_dim,
                        "entity_set": entity_set,
                        "names": io.get("names") or ["f%d" % i for i in range(obs_dim)],
                        "normalize": True},
                "action": {"space_type": space_type, "dim": action_dim},
            },
        },
        "networks": networks,
        "algo": {"name": algo_name, "config": algo_cfg},
        "train": dict(form.get("train") or {}),
    }
    if env.get("type") == "combat":
        cfg["env"]["config"] = {
            "num_enemies": int(env.get("num_enemies", 3)),
            "max_detected": int(env.get("max_detected", 10)),
            "max_episode_steps": int(env.get("max_episode_steps", 300)),
        }

    strategy = form.get("strategy") or {}
    if strategy.get("name"):
        cfg["strategy"] = {"name": strategy["name"],
                           "config": dict(strategy.get("config") or {}),
                           "extra": {}}
    return cfg, warnings


def _build_dl_config(form, name):
    io = dict(form.get("io") or {})
    obs_dim = int(io.get("obs_dim", 10))
    stack = copy.deepcopy(form.get("stack") or [])
    warnings = []
    for layer in stack:
        cfg = layer.setdefault("config", {})
        if layer.get("type", "").lower() == "mlp":
            cfg.setdefault("in_dim", obs_dim)
    data = dict(form.get("dl") or {})
    cfg = {
        "meta": {"name": name, "desc": form.get("desc", "")},
        "device": "cpu",
        "network_stack": stack,
        "io": {"obs": {"shape": [obs_dim], "dim": obs_dim, "entity_set": False}},
        "data": {
            "path": data.get("path", ""),
            "task": data.get("task", "regression"),
            "batch_size": int(data.get("batch_size", 64)),
            "val_split": float(data.get("val_split", 0.1)),
        },
        "train": {
            "seed": int((form.get("train") or {}).get("seed", 42)),
            "epochs": int(data.get("epochs", 20)),
            "lr": float(data.get("lr", 1e-3)),
        },
    }
    return cfg, warnings


def validate_config(cfg):
    """静态校验（不联网、不建网络），返回 (errors, warnings)。"""
    errs, warns = [], []
    meta = cfg.get("meta") or {}
    if not meta.get("name"):
        errs.append("配置名称为空")
    if not cfg.get("algo") and cfg.get("network_stack"):
        # 深度学习配置（dl-train）：没有 algo/env.io 的 RL 契约
        if not ((cfg.get("data") or {}).get("path")):
            errs.append("深度学习配置缺少数据文件路径 data.path")
        return errs, warns
    algo = (cfg.get("algo") or {}).get("name")
    meta_algo = ALGO_BY_NAME.get(algo)
    if meta_algo is None:
        errs.append("未知算法: %s" % algo)
    elif not meta_algo.get("ok", True):
        errs.append("%s 的后端实现尚未适配 JSON 工厂，请改用 PPO/DQN 系算法" % meta_algo["label"])

    io = ((cfg.get("env") or {}).get("io") or {})
    obs = io.get("obs") or {}
    act = io.get("action") or {}
    shape = obs.get("shape") or []
    space = act.get("space_type", "discrete")
    if len(shape) != 2:
        errs.append("观测 shape 应为 [实体数, 特征维]")
    if int(act.get("dim", 0)) <= 0:
        errs.append("动作维度必须大于 0")
    if meta_algo:
        family = meta_algo["family"]
        if family == "continuous" and space != "continuous":
            errs.append("%s 需要连续动作空间" % meta_algo["label"])
        if family != "continuous" and space != "discrete":
            errs.append("%s 需要离散动作空间" % meta_algo["label"])
        for role, spec in (cfg.get("networks") or {}).items():
            stack = (spec or {}).get("network_stack") or []
            if not stack:
                errs.append("网络角色 %s 没有网络堆叠" % role)
                continue
            if layer_mismatch(stack, int(shape[-1]) if shape else 0):
                warns.append("网络入口维度与观测特征维不一致（将由 make_env 对齐）")

    env = cfg.get("env") or {}
    if env.get("type") == "chess_remote":
        if int(env.get("action_dim", 0)) != int(act.get("dim", 0)):
            errs.append("env.action_dim 与 io.action.dim 不一致")
        obs_shape = shape
        if obs_shape and int(env.get("config", {}).get("max_entities", obs_shape[0])) != int(obs_shape[0]):
            errs.append("env.config.max_entities 与 io.obs.shape[0] 不一致")

    strategy = cfg.get("strategy") or {}
    if strategy.get("name") == "transfer_learning" \
            and not (strategy.get("config") or {}).get("pretrained"):
        errs.append("迁移学习需要填写预训练权重路径 pretrained")
    return errs, warns


def layer_mismatch(stack, obs_dim):
    first = (stack[0] or {}).get("config") or {}
    key = (stack[0] or {}).get("type", "").lower()
    if key == "mlp":
        return "in_dim" in first and int(first["in_dim"]) != int(obs_dim)
    if key in ("lstm", "gru"):
        return "input_size" in first and int(first["input_size"]) != int(obs_dim)
    return False


def dry_run(cfg):
    """真实构建网络并做一次前向（延迟 import torch，避免 GUI 启动卡顿）。"""
    import numpy as np
    from fz_agent.dl.factory import build_rl_from_json_config

    clean = copy.deepcopy(cfg)
    clean.pop("strategy", None)          # 策略包装可能需要外部 teacher/agent 列表
    agent, roles = build_rl_from_json_config(clean)
    shape = list(((clean.get("env") or {}).get("io") or {}).get("obs", {}).get("shape") or [1, 3])
    x = np.zeros(tuple(shape), dtype=np.float32)
    action = agent.act(x, deterministic=True)
    return True, ["网络构建 OK（角色: %s）" % ", ".join(roles),
                  "前向 OK（obs%s → action=%r）" % (tuple(shape), action)]


def config_signature(cfg):
    """结构签名：同样配置得到同样签名，可用于权重/结构一一对应校验。"""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


# ---------------- 训练输出解析 / 曲线 ----------------
def parse_train_line(line):
    """把训练子进程的一行输出分类，供 GUI 更新进度条。"""
    text = line.rstrip()
    low = text.lower()
    if ("构建体已导出" in text) or ("bundle" in low and "已导出" in text):
        return "bundle", text
    if "训练记录已保存" in text:
        return "saved", text
    import re
    m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", text)
    if m:
        return "epoch", (int(m.group(1)), int(m.group(2)))
    m = re.search(r"step\s+(\d+)\s*/\s*(\d+)", text)
    if m:
        return "step", (int(m.group(1)), int(m.group(2)))
    if "Traceback" in text or "Error" in text or "错误" in text:
        return "error", text
    return "log", text


def _read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def read_curves(run_dir):
    """读取一个 run 目录的 CSV，返回绘图所需的序列。"""
    epoch_rows = _read_csv(os.path.join(run_dir, "train_epoch.csv"))
    eval_rows = _read_csv(os.path.join(run_dir, "eval.csv"))
    episode_rows = _read_csv(os.path.join(run_dir, "train_episode.csv"))

    def nums(rows, key):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(key, "")))
            except (TypeError, ValueError):
                out.append(None)
        return out

    return {
        "epochs": [r.get("epoch") for r in epoch_rows],
        "train_reward": nums(epoch_rows, "train_reward_mean20"),
        "eval_reward": nums(eval_rows, "mean_reward"),
        "win_rate": nums(eval_rows, "win_rate"),
        "eval_epochs": [r.get("epoch") for r in eval_rows],
        "episode_reward": nums(episode_rows, "reward"),
    }


# ---------------- 目录扫描 ----------------
def list_configs():
    if not CONFIG_DIR.exists():
        return []
    return sorted(str(p) for p in CONFIG_DIR.glob("*.json"))


def list_bundles():
    out = []
    if not BUNDLE_DIR.exists():
        return out
    for d in sorted(BUNDLE_DIR.iterdir()):
        meta_path = d / "bundle.json"
        if d.is_dir() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            out.append({"path": str(d), "id": meta.get("bundle_id", d.name),
                        "created": meta.get("created_at", ""),
                        "algo": (meta.get("metadata") or {}).get("algorithm", ""),
                        "io": meta.get("io", {})})
    return out


def list_runs():
    out = []
    if not RUNS_DIR.exists():
        return out
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0,
                    reverse=True):
        if d.is_dir():
            out.append({"path": str(d), "name": d.name})
    return out


def list_replays(root=None):
    root = root or DEFAULT_REPLAY_DIR
    out = []
    if not os.path.isdir(root):
        return out
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(".jsonl"):
                path = os.path.join(base, name)
                try:
                    mtime = os.path.getmtime(path)
                    size = os.path.getsize(path)
                except OSError:
                    continue
                out.append({"path": path, "name": name, "mtime": mtime,
                            "size_kb": round(size / 1024.0, 1)})
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# ---------------- 子进程命令 ----------------
def training_command(config_path, python=None):
    import sys
    return [python or sys.executable, "-m", "fz_agent.main", "rl-train", "-c", config_path]


def serve_command(bundle_dir=None, host="0.0.0.0", port=8765, python=None):
    import sys
    cmd = [python or sys.executable, "-m", "fz_agent.serve",
           "--host", host, "--port", str(port)]
    cmd += ["--bundle-dir", bundle_dir or str(BUNDLE_DIR)]
    return cmd


def exp_command(bundle_path, episodes=3, python=None):
    import sys
    return [python or sys.executable, "-m", "fz_agent.main", "rl-exp",
            "--bundle", bundle_path, "--episodes", str(episodes)]


def save_config(cfg, filename=None):
    """把配置写进 fz_agent/configs/<name>.json，返回路径。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    name = filename or (cfg.get("meta", {}).get("name") or "agent")
    if not name.lower().endswith(".json"):
        name += ".json"
    path = CONFIG_DIR / name
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

if __name__ == "__main__":
    # 无界面自检：逐个预设组装 + 静态校验，便于在 WSL 里快速确认配置可用
    for _key in PRESETS:
        _cfg, _warns = build_config(dict(PRESETS[_key]))
        _errs, _more = validate_config(_cfg)
        print("[%s] errors=%s warnings=%s" % (_key, _errs, _warns + _more))

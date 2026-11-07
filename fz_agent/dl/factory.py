"""网络组装工厂：JSON 配置 → backbone / 完整 RL Agent。

支持两种网络描述方式：
1. 堆叠式 network_stack  —— 简单 Sequential 串联（当前实现）
2. 图结构 network_graph  —— DAG / 分支 / 多输入（预留接口，待实现）

两种 RL 使用模式：
A. 直接用网络：build_network(cfg) 返回 backbone
B. 完整 Agent：build_rl_from_json_config(cfg) 返回 (agent, backbone)
   后者会根据算法类型自动把 backbone 接成完整网络（如 actor/critic head）
"""

import json
import copy
import torch
import torch.nn as nn

from .blocks import build_network_by_type, NET_REGISTRY

from ..rl.ppo import PPO, ActorCritic
from ..rl.dqn import DQN, QNetwork
from ..rl import build_rl_algorithm

from ..stra.strategy_wrapper import wrap_agent_with_strategy


# ============================================================
# 网络构建（堆叠 / 图）
# ============================================================

def build_backbone_from_stack(stack_spec: list) -> nn.Module:
    """从堆叠列表构建 Sequential 主干。"""
    modules = []
    for block_spec in stack_spec:
        block = build_network_by_type(block_spec["type"], block_spec["config"])
        modules.append(block)
    return nn.Sequential(*modules) if modules else nn.Identity()


def build_backbone_from_graph(graph_spec: dict) -> nn.Module:
    """从 DAG 图结构构建网络（预留接口）。"""
    raise NotImplementedError(
        "network_graph（DAG/分支式）尚未实现，请先使用 network_stack"
    )


def build_network(raw_cfg: dict) -> nn.Module:
    """根据 JSON 配置构建 backbone（特征提取部分）。"""
    if "network_graph" in raw_cfg and raw_cfg["network_graph"]:
        return build_backbone_from_graph(raw_cfg["network_graph"])
    if "network_stack" in raw_cfg and raw_cfg["network_stack"]:
        return build_backbone_from_stack(raw_cfg["network_stack"])
    raise ValueError("JSON 缺少 'network_stack' 或 'network_graph' 字段")


# ============================================================
# Backbone 输出维度推断
# ============================================================

def _infer_backbone_out_dim(backbone: nn.Module, stack_spec: list) -> int:
    """跑一次 dummy forward 推断 backbone 输出维度。

    优先从 stack_spec 第一层 config 中读取 in_dim / input_size 等。
    """
    first_cfg = stack_spec[0]["config"] if stack_spec else {}

    # 构造 dummy 输入，尝试多种形状
    in_dim = (
        first_cfg.get("in_dim")
        or first_cfg.get("input_size")
        or first_cfg.get("in_channels", 3)
        or 16
    )

    # 3D 输入（LSTM/GRU/Conv1d 期望 [B, T, F] 或 [B, C, L]）
    if any(k in stack_spec[0]["type"].lower() for k in ("lstm", "gru")):
        dummy = torch.randn(2, 4, in_dim)          # [B, T, F]
    elif "conv1d" in stack_spec[0]["type"].lower():
        dummy = torch.randn(2, in_dim, 32)        # [B, C, L]
    elif "cnn" in stack_spec[0]["type"].lower() or "resnet" in stack_spec[0]["type"].lower():
        dummy = torch.randn(2, in_dim, 32, 32)    # [B, C, H, W]
    else:
        dummy = torch.randn(2, in_dim)            # [B, F] 默认 2D

    backbone.eval()
    with torch.no_grad():
        out = backbone(dummy)

    # 输出可能是 [B, D] / [B, T, D] / [B, C, H, W]
    if out.dim() == 2:
        return out.shape[1]
    elif out.dim() == 3:
        return out.shape[-1]
    elif out.dim() == 4:
        # CNN / ResNet：取通道数 * H * W 作为展平后的特征维度
        return out.shape[1] * out.shape[2] * out.shape[3]
    raise RuntimeError(f"无法推断 backbone 输出维度，out.shape={out.shape}")


# ============================================================
# 根据算法组装完整网络
# ============================================================

def _assemble_ppo_network(backbone: nn.Module, stack_spec: list,
                          algo_cfg: dict) -> ActorCritic:
    """用 backbone 组装 PPO 的 ActorCritic 完整网络。

    如果 stack_spec 只有一层 mlp 并且 in_dim == action_dim 场景匹配，
    或者 backbone 已经是 ActorCritic，直接返回。
    否则：backbone 之后接 masked pooling + actor/critic head。
    """
    out_dim = _infer_backbone_out_dim(backbone, stack_spec)
    action_dim = algo_cfg.get("action_dim", 3)

    # 检查 backbone 是否已经是 ActorCritic
    if isinstance(backbone, ActorCritic):
        return backbone

    class _WrappedActorCritic(nn.Module):
        def __init__(self, backbone, feat_dim, act_dim):
            super().__init__()
            self.backbone = backbone
            self.feat_dim = feat_dim

            # 推断 backbone 输出形状：如果是 [B, N, D] 做 pooling，否则直接用
            self._need_pool = None  # 懒检测

            self.shared = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.Tanh(),
            )
            self.actor = nn.Linear(feat_dim, act_dim)
            self.critic = nn.Linear(feat_dim, 1)

        def _adapted_forward(self, x):
            """把 backbone 输出适配成 [B, feat_dim]。"""
            out = self.backbone(x)
            if out.dim() == 3:
                # [B, N, D] → masked mean pooling
                valid_mask = torch.any(torch.abs(x) > 1e-8, dim=-1) if x.dim() == 3 \
                    else torch.ones(out.shape[:2], device=out.device, dtype=torch.bool)
                mask = valid_mask.unsqueeze(-1).to(out.dtype)
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                return pooled
            if out.dim() == 4:
                # [B, C, H, W] → flatten
                return out.flatten(1)
            return out

        def forward(self, obs):
            if obs.dim() == 2:
                obs = obs.unsqueeze(0)
            feat = self._adapted_forward(obs)
            shared = self.shared(feat)
            logits = self.actor(shared)
            value = self.critic(shared).squeeze(-1)
            return logits, value

        def get_action_and_value(self, obs, action=None):
            logits, value = self.forward(obs)
            from torch.distributions import Categorical
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, log_prob, entropy, value

    return _WrappedActorCritic(backbone, out_dim, action_dim)


def _assemble_dqn_network(backbone: nn.Module, stack_spec: list,
                          algo_cfg: dict) -> QNetwork:
    """用 backbone 组装 DQN 的 QNetwork 完整网络。"""
    out_dim = _infer_backbone_out_dim(backbone, stack_spec)
    action_dim = algo_cfg.get("action_dim", 3)

    if isinstance(backbone, QNetwork):
        return backbone

    class _WrappedQNet(nn.Module):
        def __init__(self, backbone, feat_dim, act_dim):
            super().__init__()
            self.backbone = backbone
            self.feat_dim = feat_dim
            self.shared = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.Tanh(),
            )
            self.q_head = nn.Linear(feat_dim, act_dim)

        def forward(self, obs):
            if obs.dim() == 2:
                obs = obs.unsqueeze(0)
            out = self.backbone(obs)
            if out.dim() == 3:
                valid_mask = torch.any(torch.abs(obs) > 1e-8, dim=-1) if obs.dim() == 3 \
                    else torch.ones(out.shape[:2], device=out.device, dtype=torch.bool)
                mask = valid_mask.unsqueeze(-1).to(out.dtype)
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
                feat = pooled
            elif out.dim() == 4:
                feat = out.flatten(1)
            else:
                feat = out
            shared = self.shared(feat)
            return self.q_head(shared)

    return _WrappedQNet(backbone, out_dim, action_dim)


# ============================================================
# 完整组装入口
# ============================================================

def build_rl_from_json_config(raw_cfg: dict):
    """核心组装入口：JSON → backbone → 完整网络 → RL Agent → (可选) strategy wrapper。

    返回 (agent, backbone)。
    """
    # Step 1: backbone
    backbone = build_network(raw_cfg)
    stack_spec = raw_cfg.get("network_stack", [])

    # Step 2: 组装完整网络（根据算法类型）
    algo_section = raw_cfg.get("algo", {})
    algo_name = algo_section.get("name", "").lower()
    algo_cfg = algo_section.get("config", {})

    if algo_name == "ppo":
        full_net = _assemble_ppo_network(backbone, stack_spec, algo_cfg)
    elif algo_name in ("dqn", "ddqn", "dueling_dqn", "rainbow"):
        full_net = _assemble_dqn_network(backbone, stack_spec, algo_cfg)
    else:
        # 其他算法直接传 backbone（预留扩展）
        full_net = backbone

    # Step 3: 构建算法
    agent = build_rl_algorithm(algo_name, algo_cfg, full_net)

    # Step 4: 可选训练策略包装
    if "strategy" in raw_cfg and raw_cfg["strategy"]:
        st = raw_cfg["strategy"]
        st_name = st.get("name", "").lower()
        st_cfg = st.get("config", {})
        st_extra = st.get("extra", {})
        if st_name:
            agent = wrap_agent_with_strategy(st_name, agent, st_cfg, **st_extra)

    return agent, backbone


# ============================================================
# 辅助
# ============================================================

def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

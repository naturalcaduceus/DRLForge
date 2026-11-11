"""网络组装工厂：统一 JSON → 多网络角色 → 完整 RL Agent。

JSON schema（两种格式均支持）：

=== 旧格式（单 backbone，向后兼容）===
{
    "network_stack": [{"type": "mlp", "config": {...}}, ...],
    "algo": {"name": "ppo", "config": {...}},
}
→ factory 自动推断角色并补 head。

=== 新格式（多网络角色，连续动作算法必需）===
{
    "env": {"action_dim": 3, "action_space_type": "discrete"},
    "networks": {
        "policy": {"network_stack": [...], "head": "actor_critic"},
        "critic": {"network_stack": [...], "head": "critic_q_continuous"},
    },
    "algo": {"name": "ddpg", "config": {...}},
}

head 类型（7 种）：
    actor_critic          → 输出 (logits, value)      PPO / A2C
    actor_discrete        → 输出 logits               REINFORCE
    q                     → 输出 Q(s,·)               DQN / DDQN
    dueling_q             → Dueling Q                  DuelingDQN
    actor_continuous      → 输出 tanh(action_mean)    DDPG / TD3
    actor_sac             → 输出 (mean, log_std)       SAC
    critic_q_continuous   → 输入 (obs, action) → Q     DDPG / TD3 / SAC
"""

import json
import copy
import torch
import torch.nn as nn

from .blocks import build_network_by_type

from ..rl.ppo import PPO
from ..rl.dqn import DQN
from ..rl.ddqn import DDQN
from ..rl.dueling_dqn import DuelingDQN
from ..rl.rainbow import Rainbow
from ..rl.a2c import A2C
from ..rl.reinforce import REINFORCE
from ..rl.ddpg import DDPG
from ..rl.td3 import TD3
from ..rl.sac import SAC
from ..rl.iqn import IQN
from ..rl import build_rl_algorithm

from ..stra.strategy_wrapper import wrap_agent_with_strategy


# ============================================================
# Backbone 构建
# ============================================================

def build_backbone_from_stack(stack_spec: list) -> nn.Module:
    modules = [build_network_by_type(s["type"], s["config"]) for s in stack_spec]
    return nn.Sequential(*modules) if modules else nn.Identity()


class GraphBackbone(nn.Module):
    """非堆叠式（DAG / 分支 / 残差连接）网络。

    JSON:
    {
      "input": {"dim": 3, "entity_set": true},
      "output": "out",
      "nodes": [
        {"id": "a", "type": "mlp", "config": {...}},
        {"id": "b", "type": "mlp", "config": {...}, "inputs": ["a"]},
        {"id": "add1", "type": "add", "inputs": ["a", "b"]},
        {"id": "out", "type": "mlp", "config": {...}, "inputs": ["add1"]}
      ]
    }

    合并节点 type ∈ {add, concat, mul}，其余 type 走 NET_REGISTRY。
    inputs 缺省时连接图输入 "__input__"。
    """

    INPUT_KEY = "__input__"
    MERGE_TYPES = {"add", "concat", "mul"}

    def __init__(self, graph_spec: dict):
        super().__init__()
        nodes = graph_spec.get("nodes", [])
        if not nodes:
            raise ValueError("network_graph.nodes 不能为空")

        modules, merge_ops, inputs, declared = {}, {}, {}, []
        for spec in nodes:
            nid = spec["id"]
            ntype = str(spec["type"]).lower()
            declared.append(nid)
            inputs[nid] = list(spec.get("inputs", [self.INPUT_KEY]))
            if ntype in self.MERGE_TYPES:
                merge_ops[nid] = ntype
            else:
                modules[nid] = build_network_by_type(spec["type"], spec.get("config", {}))

        self.output = graph_spec.get("output", declared[-1])
        if self.output not in declared:
            raise ValueError(f"network_graph.output '{self.output}' 不在 nodes 中")

        self.nodes = nn.ModuleDict(modules)
        self.merge_ops = merge_ops
        self.edges = inputs
        self.order = self._topo_order(declared)
        self.input_dim = int(graph_spec.get("input", {}).get("dim", 3))
        self.entity_set = bool(graph_spec.get("input", {}).get("entity_set", False))

    def _topo_order(self, declared):
        known = set(declared)
        order, state = [], {}

        def visit(nid):
            if state.get(nid) == 2:
                return
            if state.get(nid) == 1:
                raise ValueError(f"network_graph 存在环：{nid}")
            state[nid] = 1
            for src in self.edges[nid]:
                if src != self.INPUT_KEY:
                    if src not in known:
                        raise ValueError(f"节点 {nid} 引用了不存在的输入 {src}")
                    visit(src)
            state[nid] = 2
            order.append(nid)

        for nid in declared:
            visit(nid)
        return order

    def forward(self, x):
        memo = {self.INPUT_KEY: x}
        for nid in self.order:
            srcs = [memo[s] for s in self.edges[nid]]
            if nid in self.merge_ops:
                op = self.merge_ops[nid]
                if op == "add":
                    out = srcs[0]
                    for t in srcs[1:]:
                        out = out + t
                elif op == "mul":
                    out = srcs[0]
                    for t in srcs[1:]:
                        out = out * t
                else:  # concat
                    out = torch.cat(srcs, dim=-1)
            else:
                out = self.nodes[nid](srcs[0] if len(srcs) == 1 else srcs)
            memo[nid] = out
        return memo[self.output]


def build_backbone_from_graph(graph_spec: dict) -> nn.Module:
    return GraphBackbone(graph_spec)


def _infer_out_dim_graph(backbone: nn.Module, graph_spec: dict) -> int:
    dim = int(graph_spec.get("input", {}).get("dim", 3))
    entity_set = bool(graph_spec.get("input", {}).get("entity_set", False))
    dummy = torch.randn(2, 4, dim) if entity_set else torch.randn(2, dim)
    backbone.eval()
    with torch.no_grad():
        out = backbone(dummy)
    if out.dim() == 2:
        return out.shape[1]
    if out.dim() == 3:
        return out.shape[-1]
    if out.dim() == 4:
        return out.shape[1] * out.shape[2] * out.shape[3]
    raise RuntimeError(f"无法推断 graph backbone 输出维度，out.shape={out.shape}")


def _infer_out_dim(backbone: nn.Module, stack_spec: list) -> int:
    first_cfg = stack_spec[0]["config"] if stack_spec else {}
    in_dim = (
        first_cfg.get("in_dim")
        or first_cfg.get("input_size")
        or first_cfg.get("in_channels", 3)
        or 16
    )
    t0 = stack_spec[0]["type"].lower() if stack_spec else "mlp"
    if any(k in t0 for k in ("lstm", "gru")):
        dummy = torch.randn(2, 4, in_dim)
    elif "conv1d" in t0:
        dummy = torch.randn(2, in_dim, 32)
    elif "cnn" in t0 or "resnet" in t0:
        dummy = torch.randn(2, in_dim, 32, 32)
    else:
        dummy = torch.randn(2, in_dim)
    backbone.eval()
    with torch.no_grad():
        out = backbone(dummy)
    if out.dim() == 2:
        return out.shape[1]
    if out.dim() == 3:
        return out.shape[-1]
    if out.dim() == 4:
        return out.shape[1] * out.shape[2] * out.shape[3]
    raise RuntimeError(f"无法推断 backbone 输出维度，out.shape={out.shape}")


# ============================================================
# Head 工厂：backbone → 完整网络
# ============================================================

def _pool_feat(backbone, x):
    """3D → masked pooling, 4D → flatten, 2D → 直接用。"""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    out = backbone(x)
    if out.dim() == 3:
        valid = torch.any(torch.abs(x) > 1e-8, dim=-1)
        mask = valid.unsqueeze(-1).to(out.dtype)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    if out.dim() == 4:
        return out.flatten(1)
    return out


def _make_actor_critic_head(backbone, feat_dim, action_dim):
    class _Net(nn.Module):
        def __init__(self, bb, fd, ad):
            super().__init__()
            self.backbone = bb
            self.shared = nn.Sequential(nn.Linear(fd, fd), nn.Tanh())
            self.actor = nn.Linear(fd, ad)
            self.critic = nn.Linear(fd, 1)

        def forward(self, obs):
            f = _pool_feat(self.backbone, obs)
            s = self.shared(f)
            return self.actor(s), self.critic(s).squeeze(-1)

        def get_action_and_value(self, obs, action=None):
            logits, value = self.forward(obs)
            from torch.distributions import Categorical
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value
    return _Net(backbone, feat_dim, action_dim)


def _make_actor_discrete_head(backbone, feat_dim, action_dim):
    class _Net(nn.Module):
        def __init__(self, bb, fd, ad):
            super().__init__()
            self.backbone = bb
            self.head = nn.Sequential(nn.Linear(fd, fd), nn.Tanh(), nn.Linear(fd, ad))

        def forward(self, x):
            return self.head(_pool_feat(self.backbone, x))
    return _Net(backbone, feat_dim, action_dim)


def _make_q_head(backbone, feat_dim, action_dim):
    class _Net(nn.Module):
        def __init__(self, bb, fd, ad):
            super().__init__()
            self.backbone = bb
            self.head = nn.Sequential(nn.Linear(fd, fd), nn.Tanh(), nn.Linear(fd, ad))

        def forward(self, x):
            return self.head(_pool_feat(self.backbone, x))
    return _Net(backbone, feat_dim, action_dim)


def _make_dueling_q_head(backbone, feat_dim, action_dim):
    class _Net(nn.Module):
        def __init__(self, bb, fd, ad):
            super().__init__()
            self.backbone = bb
            self.value_stream = nn.Sequential(nn.Linear(fd, fd), nn.ReLU(), nn.Linear(fd, 1))
            self.adv_stream = nn.Sequential(nn.Linear(fd, fd), nn.ReLU(), nn.Linear(fd, ad))

        def forward(self, x):
            f = _pool_feat(self.backbone, x)
            v = self.value_stream(f)
            a = self.adv_stream(f)
            return v + a - a.mean(dim=-1, keepdim=True)
    return _Net(backbone, feat_dim, action_dim)


def _make_actor_continuous_head(backbone, feat_dim, action_dim):
    class _Net(nn.Module):
        def __init__(self, bb, fd, ad):
            super().__init__()
            self.backbone = bb
            self.head = nn.Sequential(nn.Linear(fd, fd), nn.ReLU(), nn.Linear(fd, ad), nn.Tanh())

        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out = self.backbone(x)
            if out.dim() == 3:
                out = out.mean(1)
            elif out.dim() == 4:
                out = out.flatten(1)
            return self.head(out)
    return _Net(backbone, feat_dim, action_dim)


def _make_actor_sac_head(backbone, feat_dim, action_dim):
    class _Net(nn.Module):
        def __init__(self, bb, fd, ad):
            super().__init__()
            self.backbone = bb
            self.base = nn.Sequential(nn.Linear(fd, fd), nn.ReLU())
            self.mean = nn.Linear(fd, ad)
            self.log_std = nn.Linear(fd, ad)

        def forward(self, x):
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out = self.backbone(x)
            if out.dim() == 3:
                out = out.mean(1)
            elif out.dim() == 4:
                out = out.flatten(1)
            h = self.base(out)
            return self.mean(h), self.log_std(h).clamp(-20, 2)
    return _Net(backbone, feat_dim, action_dim)


def _make_critic_q_continuous_head(backbone, feat_dim, action_dim):
    class _Net(nn.Module):
        def __init__(self, bb, fd, ad):
            super().__init__()
            self.backbone = bb
            self.head = nn.Sequential(
                nn.Linear(fd + ad, fd), nn.ReLU(),
                nn.Linear(fd, fd), nn.ReLU(),
                nn.Linear(fd, 1),
            )

        def forward(self, obs, action):
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
            out = self.backbone(obs)
            if out.dim() == 3:
                out = out.mean(1)
            elif out.dim() == 4:
                out = out.flatten(1)
            return self.head(torch.cat([out, action], dim=-1)).squeeze(-1)
    return _Net(backbone, feat_dim, action_dim)


HEAD_FACTORIES = {
    "actor_critic": _make_actor_critic_head,
    "actor_discrete": _make_actor_discrete_head,
    "q": _make_q_head,
    "dueling_q": _make_dueling_q_head,
    "actor_continuous": _make_actor_continuous_head,
    "actor_sac": _make_actor_sac_head,
    "critic_q_continuous": _make_critic_q_continuous_head,
}

# 每种算法需要的网络角色 → head 类型
ALGO_REQUIREMENTS = {
    "ppo":          {"roles": {"policy": "actor_critic"}},
    "a2c":          {"roles": {"policy": "actor_critic"}},
    "reinforce":    {"roles": {"policy": "actor_discrete"}},
    "dqn":          {"roles": {"q_net": "q"}},
    "ddqn":         {"roles": {"q_net": "q"}},
    "dueling_dqn":  {"roles": {"q_net": "dueling_q"}},
    "ddpg":         {"roles": {"actor": "actor_continuous", "critic": "critic_q_continuous"}},
    "td3":          {"roles": {"actor": "actor_continuous", "q1": "critic_q_continuous", "q2": "critic_q_continuous"}},
    "sac":          {"roles": {"actor": "actor_sac", "q1": "critic_q_continuous", "q2": "critic_q_continuous"}},
}


# ============================================================
# 组装完整网络
# ============================================================

def _build_role_networks(networks_cfg: dict, algo_name: str,
                          algo_cfg: dict) -> dict:
    req = ALGO_REQUIREMENTS.get(algo_name)
    if req is None:
        raise ValueError(f"未知算法: {algo_name}，可用: {list(ALGO_REQUIREMENTS.keys())}")

    roles_needed = req["roles"]
    action_dim = algo_cfg.get("action_dim", 3)
    built = {}

    for role_name, expected_head in roles_needed.items():
        if role_name not in networks_cfg:
            raise ValueError(
                f"算法 {algo_name} 需要网络角色 '{role_name}' (head={expected_head})，"
                f"但 networks 里没有。可用: {list(networks_cfg.keys())}"
            )

        role_cfg = networks_cfg[role_name]
        # 支持三种写法：
        #   1) 直接给 list（等价于 network_stack）
        #   2) {"network_stack": [...], "head": "..."}
        #   3) {"network_graph": {...}, "head": "..."}  ← 非堆叠式
        if isinstance(role_cfg, list):
            stack, graph, head_override = role_cfg, None, expected_head
        else:
            stack = role_cfg.get("network_stack")
            graph = role_cfg.get("network_graph")
            head_override = role_cfg.get("head", expected_head)

        if graph:
            backbone = build_backbone_from_graph(graph)
            feat_dim = _infer_out_dim_graph(backbone, graph)
        else:
            stack = stack or []
            backbone = build_backbone_from_stack(stack)
            feat_dim = _infer_out_dim(backbone, stack)

        factory = HEAD_FACTORIES.get(head_override)
        if factory is None:
            raise ValueError(f"未知 head 类型: {head_override}，可用: {list(HEAD_FACTORIES.keys())}")

        built[role_name] = factory(backbone, feat_dim, action_dim)

    return built


def _agent_kwargs_from_roles(roles: dict, algo_name: str, algo_cfg: dict) -> tuple:
    cfg_with_action = dict(algo_cfg)

    if algo_name in ("ppo", "a2c"):
        return (cfg_with_action, roles["policy"]), {}
    if algo_name == "reinforce":
        return (cfg_with_action, roles["policy"]), {}
    if algo_name in ("dqn", "ddqn"):
        import copy
        return (cfg_with_action, roles["q_net"], copy.deepcopy(roles["q_net"])), {}
    if algo_name == "dueling_dqn":
        import copy
        return (cfg_with_action, roles["q_net"], copy.deepcopy(roles["q_net"])), {}
    if algo_name == "ddpg":
        import copy
        return (cfg_with_action, roles["actor"], roles["critic"]), {}
    if algo_name == "td3":
        import copy
        return (cfg_with_action, roles["actor"], roles["q1"], roles["q2"]), {}
    if algo_name == "sac":
        import copy
        return (cfg_with_action, roles["actor"], roles["q1"], roles["q2"],
                copy.deepcopy(roles["q1"]), copy.deepcopy(roles["q2"])), {}
    raise ValueError(f"算法 {algo_name} 的 kwargs 映射未实现")


# ============================================================
# 归一化 + 顶层入口
# ============================================================

def _normalize_raw_cfg(raw_cfg: dict) -> dict:
    """把旧格式 JSON 归一化成新格式。"""
    cfg = copy.deepcopy(raw_cfg)

    if "networks" not in cfg and "network_stack" in cfg:
        algo_name = cfg.get("algo", {}).get("name", "ppo").lower()
        req = ALGO_REQUIREMENTS.get(algo_name, {})
        roles = req.get("roles", {})
        stack = cfg["network_stack"]

        # 旧格式所有角色共用同一个 stack（最简单的默认）
        networks = {}
        for role_name in roles:
            networks[role_name] = {"network_stack": stack}
        cfg["networks"] = networks
        del cfg["network_stack"]

    return cfg


def build_rl_from_json_config(raw_cfg: dict):
    """JSON → 完整 RL Agent + roles_dict。

    Returns:
        agent: 完整的 RL 算法实例
        roles: dict，如 {"policy": Net, "critic": Net}，方便外部检查/复用
    """
    cfg = _normalize_raw_cfg(raw_cfg)

    algo_section = cfg.get("algo", {})
    algo_name = algo_section.get("name", "").lower()
    algo_cfg = algo_section.get("config", {})

    roles = _build_role_networks(cfg.get("networks", {}), algo_name, algo_cfg)
    args, kwargs = _agent_kwargs_from_roles(roles, algo_name, algo_cfg)
    agent = build_rl_algorithm(algo_name, *args, **kwargs)

    if "strategy" in cfg and cfg["strategy"]:
        st = cfg["strategy"]
        st_name = st.get("name", "").lower()
        st_cfg = st.get("config", {})
        st_extra = st.get("extra", {})
        if st_name:
            agent = wrap_agent_with_strategy(st_name, agent, st_cfg, **st_extra)

    return agent, roles


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

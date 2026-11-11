"""构建体（Agent Bundle）：把"输入输出规格 + 网络结构信息 + 权重文件"
打包成一个可独立加载、可部署的单元。

一个构建体 = 
    io      : IOSpec（观测/动作契约）
    config  : 完整 JSON 配置（网络角色 + 算法 + 训练参数）
    weights : state_dict（policy/target/critic/q1/q2/...）

为什么需要它
------------
对抗循环里往往同时存在多个构建体（红方/蓝方/历史对手/教师模型），
它们各自的 io、网络结构、权重都不同。用 AgentBundle 统一封装后，
ModelPool 可以把它们同时加载进内存，SelfPlayEnv 按 id 切换对手，
InferenceService / serve.py 则可以直接对外提供推理接口。
"""
import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from .env.base import IOSpec

BUNDLE_FILE = "bundle.json"
WEIGHTS_FILE = "weights.pt"
BUNDLE_VERSION = 1

# Agent 上可能出现的网络属性（不同算法不同）
_NET_ATTRS = (
    "policy", "target",
    "critic", "target_critic",
    "q1", "q2", "target_q1", "target_q2",
)


def _unwrap(agent):
    """穿过训练策略包装器，拿到真正的 RL Agent。"""
    seen = 0
    while hasattr(agent, "agent") and seen < 8:
        agent = agent.agent
        seen += 1
    return agent


def _extract_weights(agent) -> Dict[str, dict]:
    agent = _unwrap(agent)
    weights = {}
    for attr in _NET_ATTRS:
        net = getattr(agent, attr, None)
        if isinstance(net, torch.nn.Module):
            weights[attr] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    return weights


def _load_weights(agent, weights: Dict[str, dict]) -> None:
    agent = _unwrap(agent)
    for attr, state in (weights or {}).items():
        net = getattr(agent, attr, None)
        if isinstance(net, torch.nn.Module):
            net.load_state_dict(state)
    policy = getattr(agent, "policy", None)
    if isinstance(policy, torch.nn.Module):
        policy.eval()


def _serialize_action(action):
    """把 int / tensor / ndarray 统一成可 JSON 化的值。

    单元素张量/数组统一成标量：离散动作 → int，连续动作 → float。
    否则不同算法会分别返回 1 / [1] / [1.0]，对外接口不一致。
    """
    if isinstance(action, torch.Tensor):
        a = action.detach().cpu()
        if a.numel() == 1:
            v = a.reshape(())
            return float(v.item()) if v.is_floating_point() else int(v.item())
        return a.flatten().tolist()
    if hasattr(action, "tolist"):
        data = action.tolist()
        if isinstance(data, list) and len(data) == 1 and not isinstance(data[0], (list, tuple)):
            return data[0]
        return data
    if isinstance(action, (int, float, bool)):
        return action
    return action


@dataclass
class AgentBundle:
    """一个可加载、可部署的构建体。"""

    bundle_id: str
    config: dict
    io_spec: dict
    metadata: dict = field(default_factory=dict)
    weights: Optional[Dict[str, dict]] = None
    kind: str = "rl"
    path: Optional[str] = None
    device: str = "cpu"
    _agent: Any = field(default=None, repr=False, compare=False)

    # ---------- 构建 / 推理 ----------

    def build_agent(self, apply_strategy: bool = False):
        """按 config 重建网络并载入权重。

        apply_strategy=False 时剥离 strategy 字段，只保留可部署的策略本体。
        """
        from .dl.factory import build_rl_from_json_config

        cfg = copy.deepcopy(self.config)
        if not apply_strategy:
            cfg.pop("strategy", None)
        cfg.setdefault("device", self.device)

        agent, _roles = build_rl_from_json_config(cfg)
        if self.weights:
            _load_weights(agent, self.weights)
        self._agent = agent
        return agent

    def predict(self, obs, deterministic: bool = True):
        agent = self._agent or self.build_agent()
        return _serialize_action(agent.act(obs, deterministic=deterministic))

    # ---------- 构造 ----------

    @classmethod
    def from_agent(cls, agent, config: dict, io_spec, bundle_id: str,
                   metadata: Optional[dict] = None, kind: str = "rl") -> "AgentBundle":
        io_dict = io_spec.to_dict() if isinstance(io_spec, IOSpec) else (io_spec or {})
        return cls(
            bundle_id=bundle_id,
            config=copy.deepcopy(config),
            io_spec=io_dict,
            metadata=metadata or {},
            weights=_extract_weights(agent),
            kind=kind,
            _agent=_unwrap(agent),
        )

    # ---------- 持久化 ----------

    def save(self, dirpath: str) -> str:
        os.makedirs(dirpath, exist_ok=True)
        if self._agent is not None:
            self.weights = _extract_weights(self._agent)
        torch.save(self.weights or {}, os.path.join(dirpath, WEIGHTS_FILE))

        meta = {
            "bundle_id": self.bundle_id,
            "version": BUNDLE_VERSION,
            "kind": self.kind,
            "created_at": self.metadata.get(
                "created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "io": self.io_spec,
            "weights": WEIGHTS_FILE,
            "metadata": self.metadata,
            "config": self.config,
        }
        with open(os.path.join(dirpath, BUNDLE_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        self.path = dirpath
        return dirpath

    @classmethod
    def load(cls, dirpath: str, device: str = "cpu") -> "AgentBundle":
        meta_path = os.path.join(dirpath, BUNDLE_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"构建体缺少 {BUNDLE_FILE}: {dirpath}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        weights = None
        wpath = os.path.join(dirpath, meta.get("weights", WEIGHTS_FILE))
        if os.path.exists(wpath):
            weights = torch.load(wpath, map_location="cpu")

        return cls(
            bundle_id=meta.get("bundle_id", os.path.basename(dirpath)),
            config=meta.get("config", {}),
            io_spec=meta.get("io", {}),
            metadata=meta.get("metadata", {}),
            weights=weights,
            kind=meta.get("kind", "rl"),
            path=dirpath,
            device=device,
        )


class ModelPool:
    """同时持有多个构建体，按 id 取用。"""

    def __init__(self):
        self._bundles: Dict[str, AgentBundle] = {}

    def register(self, bundle: AgentBundle) -> None:
        self._bundles[bundle.bundle_id] = bundle

    def load_dir(self, root: str, device: str = "cpu") -> List[str]:
        """加载 root 下所有含 bundle.json 的子目录（含 root 自身）。"""
        loaded = []
        candidates = []
        if os.path.exists(os.path.join(root, BUNDLE_FILE)):
            candidates.append(root)
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                sub = os.path.join(root, name)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, BUNDLE_FILE)):
                    candidates.append(sub)
        for c in candidates:
            bundle = AgentBundle.load(c, device=device)
            self.register(bundle)
            loaded.append(bundle.bundle_id)
        return loaded

    def get(self, bundle_id: str) -> AgentBundle:
        if bundle_id not in self._bundles:
            raise KeyError(f"未知构建体 '{bundle_id}'，已加载: {list(self._bundles)}")
        return self._bundles[bundle_id]

    def predict(self, bundle_id: str, obs, deterministic: bool = True):
        return self.get(bundle_id).predict(obs, deterministic=deterministic)

    def ids(self) -> List[str]:
        return list(self._bundles)

    def __contains__(self, bundle_id) -> bool:
        return bundle_id in self._bundles

    def __len__(self) -> int:
        return len(self._bundles)


class SelfPlayEnv:
    """把环境与对手构建体绑定：每一步由对手构建体给出敌方动作。

    用法：
        pool = ModelPool(); pool.load_dir("bundles")
        env = SelfPlayEnv(CombatEnvProxy(...), pool, opponent_id="blue_v1")
        env.set_opponent("blue_v2")   # 中途切换对手（联赛训练）
    """

    def __init__(self, env, pool: ModelPool, opponent_id: Optional[str] = None):
        self.env = env
        self.pool = pool
        self.opponent_id = None
        if opponent_id is not None:
            self.set_opponent(opponent_id)

    def set_opponent(self, opponent_id: Optional[str]):
        self.opponent_id = opponent_id
        if opponent_id is None:
            self.env.set_opponent(None)
        else:
            self.env.set_opponent(lambda obs: self.pool.predict(opponent_id, obs))

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        return self.env.close()

    @property
    def io_spec(self):
        return self.env.io_spec

    def __getattr__(self, item):
        return getattr(self.env, item)


class InferenceService:
    """部署侧适配：把多构建体推理包装成统一请求/响应。"""

    def __init__(self, pool: ModelPool):
        self.pool = pool

    def predict(self, bundle_id: str, obs, deterministic: bool = True) -> dict:
        action = self.pool.predict(bundle_id, obs, deterministic=deterministic)
        return {"bundle_id": bundle_id, "action": action}

    def handle(self, payload: dict) -> dict:
        """支持单请求与批量请求。

        单条： {"bundle_id": "...", "obs": [...], "deterministic": true}
        批量： {"requests": [ {...}, {...} ]}
        """
        if "requests" in payload:
            return {"responses": [self._one(r) for r in payload["requests"]]}
        return self._one(payload)

    def _one(self, req: dict) -> dict:
        try:
            return self.predict(
                req["bundle_id"],
                req["obs"],
                deterministic=bool(req.get("deterministic", True)),
            )
        except Exception as e:  # noqa: BLE001
            return {"bundle_id": req.get("bundle_id"), "error": str(e)}

    def list_models(self) -> dict:
        return {"bundles": self.pool.ids(), "count": len(self.pool)}

"""环境 I/O 契约。

这里定义"输入输出"的统一描述：观测规格 + 动作规格。
训练侧（造网络）和推理侧（喂数据）都读同一份 IOSpec，
真实仿真引擎接入时只需实现 BaseEnv 并把原始数据映射到该契约，
上层 RL / DL 代码无需改动。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ObsSpec:
    """观测规格。

    shape       观测数组形状，如 [max_detected, 3] 或 [obs_dim]
    entity_set  是否为"实体集合"型观测（[N, F]，N 可变、F 固定）
    names       每个特征维度的名字，便于前端/日志展示
    normalize   是否已归一化到 [0,1] / [-1,1]
    clip        取值裁剪范围，None 表示不裁剪
    """

    shape: List[int] = field(default_factory=lambda: [3])
    dim: int = 3
    entity_set: bool = False
    names: List[str] = field(default_factory=list)
    normalize: bool = True
    clip: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "dim": self.dim,
            "entity_set": self.entity_set,
            "names": list(self.names),
            "normalize": self.normalize,
            "clip": self.clip,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ObsSpec":
        d = d or {}
        shape = list(d.get("shape", [d.get("dim", 3)]))
        return cls(
            shape=shape,
            dim=int(d.get("dim", shape[-1])),
            entity_set=bool(d.get("entity_set", len(shape) == 2)),
            names=list(d.get("names", [])),
            normalize=bool(d.get("normalize", True)),
            clip=d.get("clip"),
        )


@dataclass
class ActionSpec:
    """动作规格。

    space_type  discrete | continuous
    dim         discrete 时为动作个数，continuous 时为动作维度
    low/high    连续动作范围（离散动作忽略）
    """

    space_type: str = "discrete"
    dim: int = 3
    low: float = -1.0
    high: float = 1.0

    @property
    def n(self) -> int:
        return self.dim

    def to_dict(self) -> dict:
        return {
            "space_type": self.space_type,
            "dim": self.dim,
            "low": self.low,
            "high": self.high,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSpec":
        d = d or {}
        return cls(
            space_type=str(d.get("space_type", "discrete")).lower(),
            dim=int(d.get("dim", d.get("action_dim", 3))),
            low=float(d.get("low", -1.0)),
            high=float(d.get("high", 1.0)),
        )


@dataclass
class IOSpec:
    """一条完整的环境输入输出契约。"""

    obs: ObsSpec = field(default_factory=ObsSpec)
    action: ActionSpec = field(default_factory=ActionSpec)

    def to_dict(self) -> dict:
        return {"obs": self.obs.to_dict(), "action": self.action.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "IOSpec":
        d = d or {}
        return cls(
            obs=ObsSpec.from_dict(d.get("obs", {})),
            action=ActionSpec.from_dict(d.get("action", d.get("action_space", {}))),
        )


class BaseEnv(ABC):
    """环境基类。

    训练/推理只依赖这一层，真实仿真引擎通过子类或 Proxy 接入。
    """

    io_spec: IOSpec

    @abstractmethod
    def reset(self, seed: Optional[int] = None):
        """返回 (obs, info)。"""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """返回 (obs, reward, terminated, truncated, info)。"""
        raise NotImplementedError

    def set_opponent(self, policy) -> None:
        """接入对手策略（对抗/自博弈）。默认空实现。"""
        raise NotImplementedError(f"{type(self).__name__} 不支持对手策略")

    def close(self) -> None:
        pass

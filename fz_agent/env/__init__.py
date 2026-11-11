"""环境包：I/O 契约 + 简易对抗环境。"""
from .base import BaseEnv, IOSpec, ObsSpec, ActionSpec
from .CombatEnv import CombatEnv, CombatEnvProxy

__all__ = [
    "BaseEnv", "IOSpec", "ObsSpec", "ActionSpec",
    "CombatEnv", "CombatEnvProxy",
]

"""向后兼容入口：转发到 .base 和 __init__.py 的注册中心。

旧代码可能 from fz_agent.rl.algos import BaseRLAlgo / build_rl_algorithm，
这里保持可用。
"""
from .base import BaseRLAlgo                # noqa: F401
from . import build_rl_algorithm            # noqa: F401
from . import ALGO_REGISTRY                 # noqa: F401

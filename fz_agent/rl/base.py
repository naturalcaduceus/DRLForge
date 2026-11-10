"""强化学习算法基类。

子类必实现:
    act(obs, deterministic=False)        → action（统一简洁接口）
    update(batch)                        → dict[str, float] 或 None

子类可选提供:
    act_with_detail(obs, deterministic)   → (action, log_prob, value...)
        用于 PPO/A2C/REINFORCE 等需要 log_prob 做 on-policy 更新的算法
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseRLAlgo(ABC):
    """所有 RL 算法的抽象基类。"""

    def __init__(self, config: dict, policy_net: nn.Module,
                 target_net: Optional[nn.Module] = None):
        self.cfg = config
        self.policy = policy_net
        self.target = target_net

        request_device = config.get("device", "cuda")
        if request_device == "cuda" and not torch.cuda.is_available():
            print("[rl] CUDA 不可用，自动回退 CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(request_device)

        self.policy.to(self.device)
        if self.target is not None:
            self.target.to(self.device)

    # ------- 统一动作选择 -------

    @abstractmethod
    def act(self, obs, deterministic=False):
        """统一简洁接口：只返回 action。

        离散动作算法（PPO/DQN/A2C/REINFORCE）→ int 或 tensor
        连续动作算法（DDPG/SAC/TD3）       → tensor (action_dim,)
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_action(self, obs, deterministic=False):
        """评估时用（no_grad 包装）。"""
        return self.act(obs, deterministic=deterministic)

    # ------- 可选：获取 value（ActorCritic 架构才有意义）-------

    @torch.no_grad()
    def get_value(self, obs):
        try:
            self.policy.eval()
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if t.dim() == 2:
                t = t.unsqueeze(0)
            with torch.no_grad():
                out = self.policy(t)
            if isinstance(out, tuple) and len(out) >= 2:
                return float(out[1].squeeze(-1).item())
        except Exception:
            pass
        return 0.0

    # ------- 基类接口（兼容旧代码）-------

    def select_action(self, obs: torch.Tensor):
        """兼容旧调用方。"""
        return self.act(obs, deterministic=False)

    @abstractmethod
    def update(self, batch: Any):
        """训练更新。"""
        raise NotImplementedError

    # ------- 保存/加载 -------

    def save_checkpoint(self, path: str):
        state = {"policy": self.policy.state_dict(), "cfg": self.cfg}
        if self.target is not None:
            state["target"] = self.target.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        if self.target is not None and ckpt.get("target") is not None:
            self.target.load_state_dict(ckpt["target"])

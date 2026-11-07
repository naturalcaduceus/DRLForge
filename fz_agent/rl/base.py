"""强化学习算法基类。"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseRLAlgo(ABC):
    """所有 RL 算法的抽象基类。

    子类需实现 select_action 和 update。
    """

    def __init__(self, config: dict, policy_net: nn.Module,
                 target_net: Optional[nn.Module] = None):
        self.cfg = config
        self.policy = policy_net
        self.target = target_net

        # 设备容错：请求 CUDA 但不可用时自动回退 CPU
        request_device = config.get("device", "cuda")
        if request_device == "cuda" and not torch.cuda.is_available():
            print("[rl] CUDA 不可用，自动回退 CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(request_device)

        self.policy.to(self.device)
        if self.target is not None:
            self.target.to(self.device)

    @abstractmethod
    def select_action(self, obs: torch.Tensor):
        """选择动作。子类签名可能不同（DQN 返回 tensor，PPO 返回 (action, log_prob)）。"""
        pass

    @abstractmethod
    def update(self, batch: Dict):
        """训练更新。"""
        pass

    # ------- 通用保存/加载 -------

    def save_checkpoint(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict() if self.target else None,
            "cfg": self.cfg,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        if self.target is not None and ckpt.get("target") is not None:
            self.target.load_state_dict(ckpt["target"])

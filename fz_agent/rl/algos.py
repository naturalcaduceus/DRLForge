import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional



class BaseRLAlgo(ABC):
    def __init__(self, config: dict, policy_net: nn.Module, target_net: Optional[nn.Module] = None):
        self.cfg = config
        self.policy = policy_net
        self.target = target_net

        # 原来的：
        # self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        # 替换成容错版本：
        request_device = config.get("device", "cuda")
        if request_device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA请求失败，无可用NVIDIA驱动/GPU，自动回退到CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(request_device)

        if self.target is not None:
            self.target.to(self.device)

    @abstractmethod
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, batch: Dict):
        """训练更新逻辑，这里预留，后续填充"""
        pass

    def save_checkpoint(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "target": self.target.state_dict() if self.target else None,
            "cfg": self.cfg
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        if self.target and ckpt["target"]:
            self.target.load_state_dict(ckpt["target"])


# 1
class DQN(BaseRLAlgo):
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: epsilon-greedy
        return torch.tensor(0)

    def update(self, batch: Dict):
        # TODO: DQN loss, target sync
        pass

# 2
class DDQN(BaseRLAlgo):
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO
        return torch.tensor(0)

    def update(self, batch: Dict):
        # TODO: Double DQN, action select from policy, eval from target
        pass

# 3
class DuelingDQN(BaseRLAlgo):
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO
        return torch.tensor(0)

    def update(self, batch: Dict):
        # TODO: Dueling advantage + value branch loss
        pass

# 4
class Rainbow(BaseRLAlgo):
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO
        return torch.tensor(0)

    def update(self, batch: Dict):
        # TODO: Rainbow 6改进：优先回放、多步、C51、Noisy、Dueling、Double
        pass

# 5
class DDPG(BaseRLAlgo):
    def __init__(self, config: dict, actor: nn.Module, critic: nn.Module, target_actor=None, target_critic=None):
        super().__init__(config, actor, target_actor)
        self.critic = critic
        self.target_critic = target_critic
        self.critic.to(self.device)
        if self.target_critic:
            self.target_critic.to(self.device)

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: actor + exploration noise
        return torch.tensor(0.0)

    def update(self, batch: Dict):
        # TODO: DDPG actor-critic soft update
        pass

# 6
class PPO(BaseRLAlgo):
    def __init__(self, config: dict, actor_critic: nn.Module):
        super().__init__(config, actor_critic)

    def select_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: action, logprob, value
        return torch.tensor(0), torch.tensor(0.0)

    def update(self, rollout: Dict):
        # TODO: PPO clip loss, value loss, entropy
        pass

# 7
class SAC(BaseRLAlgo):
    def __init__(self, config: dict, actor, q1, q2, target_q1, target_q2):
        super().__init__(config, actor)
        self.q1 = q1
        self.q2 = q2
        self.target_q1 = target_q1
        self.target_q2 = target_q2

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: reparameterization
        return torch.tensor(0.0)

    def update(self, batch: Dict):
        # TODO: SAC entropy RL
        pass

# 8
class TD3(BaseRLAlgo):
    def __init__(self, config: dict, actor, q1, q2, target_actor, target_q1, target_q2):
        super().__init__(config, actor, target_actor)
        self.q1 = q1
        self.q2 = q2
        self.target_q1 = target_q1
        self.target_q2 = target_q2

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: target policy smoothing noise
        return torch.tensor(0.0)

    def update(self, batch: Dict):
        # TODO: TD3 twin critic, delayed actor update
        pass

# 9
class A2C(BaseRLAlgo):
    def __init__(self, config: dict, actor_critic: nn.Module):
        super().__init__(config, actor_critic)

    def select_action(self, obs: torch.Tensor):
        # TODO
        return torch.tensor(0), torch.tensor(0.0)

    def update(self, rollout: Dict):
        # TODO: advantage actor critic
        pass

#10
class REINFORCE(BaseRLAlgo):
    def __init__(self, config: dict, policy_net: nn.Module):
        super().__init__(config, policy_net)

    def select_action(self, obs: torch.Tensor):
        # TODO
        return torch.tensor(0), torch.tensor(0.0)

    def update(self, episode_trajectory: Dict):
        # TODO: reinforce monte carlo
        pass

#11
class IQN(BaseRLAlgo):
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # TODO: Implicit Quantile Network
        return torch.tensor(0)

    def update(self, batch: Dict):
        # TODO: quantile regression loss
        pass


ALGO_REGISTRY = {
    "dqn": DQN,
    "ddqn": DDQN,
    "dueling_dqn": DuelingDQN,
    "rainbow": Rainbow,
    "ddpg": DDPG,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "a2c": A2C,
    "reinforce": REINFORCE,
    "iqn": IQN
}

def build_rl_algorithm(algo_name: str, cfg: dict, *args, **kwargs):
    cls = ALGO_REGISTRY[algo_name.lower()]
    return cls(cfg, *args, **kwargs)

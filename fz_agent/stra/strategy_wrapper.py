"""训练策略包装器(演化训练)。

六类演化训练的落地映射(这是你没想好的部分)
核心思路：每种方法 = 对 Trainer 或 Env 的一层包装(wrapper),挂在训练流水线的一个固定钩子上：
训练方法 	实现位置 	本质 
迁移学习	Trainer 初始化钩子	加载预训练权重 + 层冻结配置,就是初始化参数
对抗训练	Env 包装器	self-play / 对手池(league),轮流把历史 checkpoint 塞进环境当对手
增量持续训练	持久化钩子	checkpoint + replay buffer 落盘,断点续训；可选渐进式网络扩层
模型蒸馏	Loss 钩子	teacher 模型 forward 出 soft label,加 KL 辅助损失
多任务联合	网络/损失侧	共享 backbone + 多任务头,各头 loss 加权求和
集成投票	推理侧封装	N 个 agent 各自推理,动作空间上投票/置信度融合

也就是说训练流水线预留四个钩子：init_hook(迁移)、env_hook(对抗)、loss_hook(蒸馏、多任务)、
inference_hook(集成),六个要求全部对上号,而且后续再加新方法只是加一个 wrapper 类。

包裹原始 RL Agent,提供 before/after 钩子用于迁移学习、对抗训练、蒸馏等。
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional

from ..rl.base import BaseRLAlgo


class BaseTrainStrategy(ABC):
    """演化训练策略基类。"""

    def __init__(self, rl_agent: BaseRLAlgo, strategy_cfg: dict):
        self.agent = rl_agent
        self.strategy_cfg = strategy_cfg

    @abstractmethod
    def before_train_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def after_train_step(self, *args, **kwargs):
        pass

    def select_action(self, obs):
        return self.agent.select_action(obs)

    def update(self, batch):
        self.before_train_step(batch=batch)
        self.agent.update(batch)
        self.after_train_step(batch=batch)

    # 透传常用属性
    @property
    def device(self):
        return self.agent.device

    @property
    def policy(self):
        return self.agent.policy


# ------- 具体策略（骨架）-------

class TransferLearningWrapper(BaseTrainStrategy):
    """迁移学习：加载预训练权重、冻结部分层。"""

    def before_train_step(self, *args, **kwargs):
        # TODO: 加载预训练权重、冻结 backbone 前 n 层
        pass

    def after_train_step(self, *args, **kwargs):
        pass


class AdversarialTrainWrapper(BaseTrainStrategy):
    """对抗训练：在观测上注入对抗扰动。"""

    def before_train_step(self, *args, **kwargs):
        # TODO: 生成对抗样本并替换 batch 观测
        pass

    def after_train_step(self, *args, **kwargs):
        pass


class ContinualIncrementalWrapper(BaseTrainStrategy):
    """增量持续训练：EWC / replay buffer 防灾难性遗忘。"""

    def before_train_step(self, *args, **kwargs):
        pass

    def after_train_step(self, *args, **kwargs):
        pass


class DistillLightWrapper(BaseTrainStrategy):
    """模型蒸馏轻量化：teacher-student 蒸馏。"""

    def __init__(self, rl_agent, strategy_cfg, teacher_model: Optional[nn.Module] = None):
        super().__init__(rl_agent, strategy_cfg)
        self.teacher = teacher_model
        self.temp = strategy_cfg.get("temperature", 2.0)
        self.alpha = strategy_cfg.get("alpha", 0.7)

    def before_train_step(self, *args, **kwargs):
        pass

    def after_train_step(self, *args, **kwargs):
        pass


class MultiTaskJointWrapper(BaseTrainStrategy):
    """多任务联合训练：共享 backbone + 多 head。"""

    def before_train_step(self, *args, **kwargs):
        pass

    def after_train_step(self, *args, **kwargs):
        pass


class EnsembleVoteWrapper(BaseTrainStrategy):
    """集成投票：多个 agent 集成，动作投票 / softmax 融合。"""

    def __init__(self, rl_agent, strategy_cfg, agent_list: Optional[list] = None):
        super().__init__(rl_agent, strategy_cfg)
        self.ensemble_agents = agent_list or []

    def select_action(self, obs):
        acts = [a.select_action(obs) for a in self.ensemble_agents]
        # 投票：取众数
        if acts and isinstance(acts[0], (int, float)):
            from collections import Counter
            return Counter(int(a) for a in acts).most_common(1)[0][0]
        return self.agent.select_action(obs)

    def before_train_step(self, *args, **kwargs):
        pass

    def after_train_step(self, *args, **kwargs):
        pass


STRATEGY_REGISTRY = {
    "transfer_learning": TransferLearningWrapper,
    "adversarial_train": AdversarialTrainWrapper,
    "incremental_continual": ContinualIncrementalWrapper,
    "distill_light": DistillLightWrapper,
    "multitask_joint": MultiTaskJointWrapper,
    "ensemble_vote": EnsembleVoteWrapper,
}


def wrap_agent_with_strategy(strategy_name: str, agent: BaseRLAlgo,
                              strategy_cfg: dict, **extra):
    """按名字构建训练策略包装器。"""
    key = strategy_name.lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"未知策略 '{strategy_name}'，可用: {list(STRATEGY_REGISTRY.keys())}"
        )
    cls = STRATEGY_REGISTRY[key]
    return cls(agent, strategy_cfg, **extra)

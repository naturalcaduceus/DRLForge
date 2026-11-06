import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from rl_algos import BaseRLAlgo


class BaseTrainStrategy(ABC):
    """演化训练策略包装器，包裹原始RL算法"""
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
        # 透传
        return self.agent.select_action(obs)

    def update(self, batch):
        # 标准钩子：前置逻辑 -> 原生更新 -> 后置逻辑
        self.before_train_step(batch=batch)
        self.agent.update(batch)
        self.after_train_step(batch=batch)


class TransferLearningWrapper(BaseTrainStrategy):
    """迁移学习：加载预训练主干，冻结/微调控制"""
    def before_train_step(self, *args, **kwargs):
        # TODO: 加载预训练权重、设置requires_grad、主干冻结
        pass

    def after_train_step(self, *args, **kwargs):
        pass


class AdversarialTrainWrapper(BaseTrainStrategy):
    """对抗训练：obs加对抗扰动，robust RL"""
    def before_train_step(self, *args, **kwargs):
        # TODO: 生成对抗样本、替换batch观测
        pass

    def after_train_step(self, *args, **kwargs):
        pass


class ContinualIncrementalWrapper(BaseTrainStrategy):
    """增量持续训练：task sequence、EWC / replay buffer 防灾难性遗忘"""
    def before_train_step(self, *args, **kwargs):
        # TODO: task switch、EWC penalty 注入loss
        pass

    def after_train_step(self, *args, **kwargs):
        # TODO: 保存task参数、样本回放池管理
        pass


class DistillLightWrapper(BaseTrainStrategy):
    """模型蒸馏轻量化：teacher-student蒸馏"""
    def __init__(self, rl_agent, strategy_cfg, teacher_model: nn.Module):
        super().__init__(rl_agent, strategy_cfg)
        self.teacher = teacher_model
        self.temp = strategy_cfg.get("temperature", 2.0)

    def before_train_step(self, *args, **kwargs):
        # TODO: teacher infer soft label
        pass

    def after_train_step(self, *args, **kwargs):
        # TODO: 蒸馏loss叠加到学生更新
        pass


class MultiTaskJointWrapper(BaseTrainStrategy):
    """多任务联合训练：共享backbone + 多head，多reward信号"""
    def before_train_step(self, *args, **kwargs):
        # TODO: 区分不同task分支、动态权重
        pass

    def after_train_step(self, *args, **kwargs):
        pass


class EnsembleVoteWrapper(BaseTrainStrategy):
    """集成投票：多个agent集成，动作投票/均值融合"""
    def __init__(self, rl_agent, strategy_cfg, agent_list: list):
        super().__init__(rl_agent, strategy_cfg)
        self.ensemble_agents = agent_list

    def select_action(self, obs):
        # TODO: 集成策略，投票、softmax融合
        acts = [a.select_action(obs) for a in self.ensemble_agents]
        return torch.stack(acts).mean(dim=0)

    def before_train_step(self, *args, **kwargs):
        # TODO: 独立更新每个成员
        pass

    def after_train_step(self, *args, **kwargs):
        pass

# 类似函数指针的操作
STRATEGY_REGISTRY = {
    "transfer_learning": TransferLearningWrapper,
    "adversarial_train": AdversarialTrainWrapper,
    "incremental_continual": ContinualIncrementalWrapper,
    "distill_light": DistillLightWrapper,
    "multitask_joint": MultiTaskJointWrapper,
    "ensemble_vote": EnsembleVoteWrapper
}

def wrap_agent_with_strategy(strategy_name: str, agent: BaseRLAlgo, strategy_cfg: dict, **extra):
    cls = STRATEGY_REGISTRY[strategy_name.lower()]
    return cls(agent, strategy_cfg, **extra)

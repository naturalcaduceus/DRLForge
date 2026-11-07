"""深度学习网络积木注册中心。

每种网络一个独立文件，在这里统一注册。
"""

from .mlp import MLP
from .cnn import CNN
from .conv1d import Conv1dEncoder
from .lstm import LSTMBlock
from .gru import GRUBlock
from .attention import SelfAttentionBlock
from .transformer import TransformerBlock
from .resnet import ResNetBlock

# 注册表 key = JSON 里 net_type 的小写形式
NET_REGISTRY = {
    "mlp": MLP,
    "cnn": CNN,
    "conv1d": Conv1dEncoder,
    "lstm": LSTMBlock,
    "gru": GRUBlock,
    "attention": SelfAttentionBlock,
    "transformer": TransformerBlock,
    "resnet": ResNetBlock,
}

# 兼容旧名称（Tk 前端可能用的别名）
NET_REGISTRY_ALIAS = {
    "selfattention": "attention",
    "transformerblock": "transformer",
    "resblock": "resnet",
    "conv2d": "cnn",
}


def build_network_by_type(net_type: str, cfg: dict):
    """根据类型名构建网络块。"""
    key = net_type.lower()
    # 别名查找
    if key not in NET_REGISTRY and key in NET_REGISTRY_ALIAS:
        key = NET_REGISTRY_ALIAS[key]
    if key not in NET_REGISTRY:
        raise ValueError(
            f"未知网络类型 '{net_type}'，可用: {list(NET_REGISTRY.keys())}"
        )
    cls = NET_REGISTRY[key]
    return cls(cfg)

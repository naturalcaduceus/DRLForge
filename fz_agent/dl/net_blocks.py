# 注册中心统一管理 8 种网络

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """基础多层感知机"""
    def __init__(self, config: dict):
        super().__init__()
        in_dim = config["in_dim"]
        hidden_dims = config["hidden_dims"]
        out_dim = config["out_dim"]
        act = getattr(nn, config.get("activation", "ReLU"))
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def build_from_config(cfg):
        return MLP(cfg)


class CNN(nn.Module):
    """2D CNN 特征提取器"""
    def __init__(self, config: dict):
        super().__init__()
        channels = config["channels"]
        kernel_sizes = config["kernel_sizes"]
        strides = config["strides"]
        padding = config.get("padding", 0)
        act = getattr(nn, config.get("activation", "ReLU"))
        layers = []
        in_c = config["in_channels"]
        for oc, k, s in zip(channels, kernel_sizes, strides):
            layers.append(nn.Conv2d(in_c, oc, k, stride=s, padding=padding))
            layers.append(act())
            in_c = oc
        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_seq(x)

    @staticmethod
    def build_from_config(cfg):
        return CNN(cfg)


class Conv1dEncoder(nn.Module):
    """一维卷积时序编码器"""
    def __init__(self, config: dict):
        super().__init__()
        in_c = config["in_channels"]
        channels = config["channels"]
        kernel_sizes = config["kernel_sizes"]
        strides = config["strides"]
        act = getattr(nn, config.get("activation", "ReLU"))
        layers = []
        for oc, k, s in zip(channels, kernel_sizes, strides):
            layers.append(nn.Conv1d(in_c, oc, k, stride=s, padding="same"))
            layers.append(act())
            in_c = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def build_from_config(cfg):
        return Conv1dEncoder(cfg)


class LSTMBlock(nn.Module):
    """LSTM时序模块"""
    def __init__(self, config: dict):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=config.get("batch_first", True),
            bidirectional=config.get("bidirectional", False)
        )
        self.out_proj = nn.Linear(
            config["hidden_size"] * (2 if config.get("bidirectional") else 1),
            config["out_dim"]
        )

    def forward(self, x, hc=None):
        out, (h, c) = self.lstm(x, hc)
        return self.out_proj(out), (h, c)

    @staticmethod
    def build_from_config(cfg):
        return LSTMBlock(cfg)


class GRUBlock(nn.Module):
    """GRU时序模块"""
    def __init__(self, config: dict):
        super().__init__()
        self.gru = nn.GRU(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=config.get("batch_first", True),
            bidirectional=config.get("bidirectional", False)
        )
        self.out_proj = nn.Linear(
            config["hidden_size"] * (2 if config.get("bidirectional") else 1),
            config["out_dim"]
        )

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        return self.out_proj(out), h

    @staticmethod
    def build_from_config(cfg):
        return GRUBlock(cfg)


class SelfAttentionBlock(nn.Module):
    """多头自注意力（独立Attention模块）"""
    def __init__(self, config: dict):
        super().__init__()
        embed_dim = config["embed_dim"]
        num_heads = config["num_heads"]
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)

    @staticmethod
    def build_from_config(cfg):
        return SelfAttentionBlock(cfg)


class TransformerBlock(nn.Module):
    """Transformer Encoder 块"""
    def __init__(self, config: dict):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            batch_first=True,
            dropout=config.get("dropout", 0.1)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])

    def forward(self, x, mask=None):
        return self.encoder(x, mask=mask)

    @staticmethod
    def build_from_config(cfg):
        return TransformerBlock(cfg)


class ResNetBlock(nn.Module):
    """残差块 ResNet"""
    def __init__(self, config: dict):
        super().__init__()
        in_c = config["in_channels"]
        out_c = config["out_channels"]
        stride = config.get("stride", 1)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

    @staticmethod
    def build_from_config(cfg):
        return ResNetBlock(cfg)


NET_REGISTRY = {
    "mlp": MLP,
    "cnn": CNN,
    "conv1d": Conv1dEncoder,
    "lstm": LSTMBlock,
    "gru": GRUBlock,
    "attention": SelfAttentionBlock,
    "transformer": TransformerBlock,
    "resnet": ResNetBlock
}

def build_network_by_type(net_type: str, cfg: dict):
    cls = NET_REGISTRY[net_type.lower()]
    return cls.build_from_config(cfg)

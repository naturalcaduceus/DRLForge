"""Conv1d 一维卷积时序编码器。"""
import torch.nn as nn


class Conv1dEncoder(nn.Module):
    """一维卷积时序编码器。

    config 字段：
        in_channels  (int): 输入通道数
        channels     (list): 输出通道数列表
        kernel_sizes (list): 卷积核大小列表
        strides      (list): 步长列表
        activation   (str):  激活函数类名
    """

    def __init__(self, config: dict):
        super().__init__()
        in_c = config["in_channels"]
        channels = config["channels"]
        kernel_sizes = config["kernel_sizes"]
        strides = config["strides"]
        act_cls = getattr(nn, config.get("activation", "ReLU"))
        layers = []
        for oc, k, s in zip(channels, kernel_sizes, strides):
            layers.append(nn.Conv1d(in_c, oc, k, stride=s, padding="same"))
            layers.append(act_cls())
            in_c = oc
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

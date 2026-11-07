"""CNN 2D 卷积特征提取器。"""
import torch.nn as nn


class CNN(nn.Module):
    """2D CNN 特征提取器。

    config 字段：
        in_channels  (int): 输入通道数
        channels     (list): 输出通道数列表
        kernel_sizes (list): 卷积核大小列表
        strides      (list): 步长列表
        padding      (int):  填充，默认 0
        activation   (str):  激活函数类名
    """

    def __init__(self, config: dict):
        super().__init__()
        in_c = config["in_channels"]
        channels = config["channels"]
        kernel_sizes = config["kernel_sizes"]
        strides = config["strides"]
        padding = config.get("padding", 0)
        act_cls = getattr(nn, config.get("activation", "ReLU"))
        layers = []
        for oc, k, s in zip(channels, kernel_sizes, strides):
            layers.append(nn.Conv2d(in_c, oc, k, stride=s, padding=padding))
            layers.append(act_cls())
            in_c = oc
        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_seq(x)

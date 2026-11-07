"""MLP 多层感知机网络积木。"""
import torch.nn as nn


class MLP(nn.Module):
    """基础多层感知机。

    config 字段：
        in_dim      (int):  输入维度
        hidden_dims (list): 各隐藏层维度列表
        out_dim     (int):  输出维度
        activation  (str):  激活函数类名，默认 "ReLU"
    """

    def __init__(self, config: dict):
        super().__init__()
        in_dim = config["in_dim"]
        hidden_dims = config["hidden_dims"]
        out_dim = config["out_dim"]
        act_cls = getattr(nn, config.get("activation", "ReLU"))
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

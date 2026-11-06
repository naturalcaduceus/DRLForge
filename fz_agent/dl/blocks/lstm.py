"""LSTM 时序模块。"""
import torch.nn as nn


class LSTMBlock(nn.Module):
    """LSTM 时序模块，带线性投影输出。

    config 字段：
        input_size    (int):
        hidden_size   (int):
        num_layers    (int):
        out_dim       (int):  线性投影输出维度
        batch_first   (bool): 默认 True
        bidirectional (bool): 默认 False
    """

    def __init__(self, config: dict):
        super().__init__()
        bidirectional = config.get("bidirectional", False)
        self.lstm = nn.LSTM(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=config.get("batch_first", True),
            bidirectional=bidirectional,
        )
        self.out_proj = nn.Linear(
            config["hidden_size"] * (2 if bidirectional else 1),
            config["out_dim"],
        )

    def forward(self, x, hc=None, return_hidden=False):
        """前向传播。默认只返回输出（兼容 Sequential 堆叠）。"""
        out, hidden = self.lstm(x, hc)
        proj = self.out_proj(out)
        if return_hidden:
            return proj, hidden
        return proj

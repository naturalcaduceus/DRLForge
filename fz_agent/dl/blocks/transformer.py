"""Transformer Encoder 块。"""
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Transformer Encoder 块。

    config 字段：
        d_model           (int):
        nhead             (int):
        dim_feedforward   (int):
        num_layers        (int):
        dropout           (float): 默认 0.1
    """

    def __init__(self, config: dict):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            batch_first=True,
            dropout=config.get("dropout", 0.1),
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_layers"],
        )

    def forward(self, x, mask=None):
        return self.encoder(x, mask=mask)

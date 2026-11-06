"""SelfAttention 多头自注意力模块。"""
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """多头自注意力 + LayerNorm 残差。

    config 字段：
        embed_dim (int):
        num_heads (int):
    """

    def __init__(self, config: dict):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            config["embed_dim"],
            config["num_heads"],
            batch_first=True,
        )
        self.norm = nn.LayerNorm(config["embed_dim"])

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, attn_mask=mask)
        return self.norm(x + attn_out)

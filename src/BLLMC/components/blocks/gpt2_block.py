from BLLMC.components.layers.feedforward import FeedForward
from BLLMC.components.attention.multi_head import MultiHeadAttention
from BLLMC.components.layers.normalization import LayerNorm
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = LayerNorm(config.emb_dim)
        self.ln2 = LayerNorm(config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.ln1(x)
        x = self.mha(x)
        x = self.dropout(x)
        x = shortcut + x
        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = shortcut + x
        return x

    def __str__(self):
        return "TransformerBlock(mha={}, ffn={}, ln1={}, ln2={}, dropout={})".format(
            self.mha, self.ffn, self.ln1, self.ln2, self.dropout
        )

    def __repr__(self):
        return self.__str__()

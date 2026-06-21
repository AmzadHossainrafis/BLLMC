import torch
import torch.nn as nn
from BLLMC.components.attention.grouped_query import GroupedQueryAttention
from BLLMC.components.attention.multi_head import MultiHeadAttentionWithRoPE
from BLLMC.components.layers.feedforward import Llama2FeedForward
from BLLMC.components.layers.normalization import RMSNorm


class Llama3Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = RMSNorm(config.emb_dim)
        self.norm2 = RMSNorm(config.emb_dim)
        self.attn = GroupedQueryAttention(config)
        self.feed_forward = Llama2FeedForward(config)  # same as llama 3

    def forward(self, x: torch.Tensor, use_cache=False):
        h = self.norm1(x)
        x = x + self.attn(h, use_cache=use_cache)
        h = self.norm2(x)
        x = x + self.feed_forward(h)
        return x

    def reset_cache(self):
        if hasattr(self.attn, "reset_cache"):
            self.attn.reset_cache()

    def __str__(self):
        return (
            f"Llama3Block(emb_dim={self.config.emb_dim}, n_heads={self.config.n_heads}, "
            f"context_length={self.config.context_length}, drop_rate={self.config.drop_rate})"
        )


class Llama2Block(nn.Module):
    """
    Llama2-style decoder model.

    Architecture:
        Token Embedding → [Llama2Block x n_layers] → RMSNorm → LM Head

    Each Llama2Block contains:
        - Pre-RMSNorm → MultiHeadAttentionWithRoPE → Residual
        - Pre-RMSNorm → FeedForward → Residual
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = RMSNorm(config.emb_dim)
        self.norm2 = RMSNorm(config.emb_dim)
        self.attn = MultiHeadAttentionWithRoPE(config)
        self.feed_forward = Llama2FeedForward(config)

    def forward(self, x, use_cache=False):

        h = self.norm1(x)
        x = x + self.attn(h, use_cache=use_cache)
        h = self.norm2(x)
        x = x + self.feed_forward(h)
        return x

    def reset_cache(self):
        if hasattr(self.attn, "reset_cache"):
            self.attn.reset_cache()

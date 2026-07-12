# src/BLLMC/components/blocks/gptoss_block.py
import torch
import torch.nn as nn
from BLLMC.components.attention.gptoss_attention import GPTOssAttention
from BLLMC.components.layers.normalization import RMSNorm
from BLLMC.components.layers.feedforward import GPTOssFeedForward


class GPTOssBlock(nn.Module):
    """
    GPT-OSS Transformer Block.

    Structure:
        x = x + Attention(RMSNorm(x))  [Alternating sliding window size]
        x = x + MLP(RMSNorm(x))        [GPT-OSS MoE FeedForward with SwiGLU]
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(config.emb_dim)

        # Alternating sliding window (applied only to every other layer)
        sliding_window = config.sliding_window_size if layer_idx % 2 == 0 else None
        self.attn = GPTOssAttention(config, sliding_window_size=sliding_window)

        self.norm2 = RMSNorm(config.emb_dim)
        self.mlp = GPTOssFeedForward(config)

    def forward(self, x: torch.Tensor, use_cache=False):
        h = self.norm1(x)
        x = x + self.attn(h, use_cache=use_cache)
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x

    def reset_cache(self):
        if hasattr(self.attn, "reset_cache"):
            self.attn.reset_cache()

    def __str__(self):
        return (
            f"GPTOssBlock(layer_idx={self.layer_idx}, attn={self.attn}, mlp={self.mlp})"
        )

    def __repr__(self):
        return self.__str__()

"""
licence : mit
author : amzad hossain rafi
email : [EMAIL_ADDRESS]

change log :
    17-5-2026 : start
    17-5-2026 : implement layer normalization

to do :
    1. implement test cases

referance :
    this implementaion of layer normalization is same as nn.LayerNorm

"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization module.

    This module applies layer normalization to the input tensor.

    Args:
        emb_dim (int): Embedding dimension.

    Formula :
        LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)
        output = scale * LayerNorm(x) + shift
    """

    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        denom = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / denom) * self.weight


"""
licence : mit
author : amzad hossain rafi
email : amzad.rafi@northsouth.edu

change log :
    17-5-2026 : start
    17-5-2026 : implement multi head attention
    21-5-2026 : make it memory optimized
    21-5-2026 : implement k cache and v cache

to do :
    1. Implement test cases
    2. Implement multi head group
    3. Implement Multi Query Attention
    4. Implement FlashAttention

"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    This module implements the standard multi-head self-attention mechanism,
    projecting the input into multiple query, key, and value representations.
    It applies causal masking to prevent attending to future tokens.

    Args:
        config: A configuration object must contain:
            - emb_dim (int): Embedding dimension (used for both input and output projections).
            - n_heads (int): Number of attention heads. emb_dim must be divisible by n_heads.
            - context_length (int): Maximum sequence length for the causal mask.
            - drop_rate (float): Dropout probability applied to attention weights.
    """

    def __init__(self, config):
        """
        Initializes the MultiHeadAttention module.

        Args:
            config: A configuration object must contain:
                - emb_dim (int): Embedding dimension.
                - n_heads (int): Number of attention heads.
                - context_length (int): Context window size for masking.
                - drop_rate (float): Dropout rate.
        """
        super().__init__()
        assert config.emb_dim % config.n_heads == 0

        self.d_out = config.emb_dim
        self.n_head = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads

        ## layere
        self.wq = nn.Linear(config.emb_dim, self.d_out, bias=False)
        self.wk = nn.Linear(config.emb_dim, self.d_out, bias=False)
        self.wv = nn.Linear(config.emb_dim, self.d_out, bias=False)

        self.out_proj = nn.Linear(self.d_out, self.d_out, bias=False)
        self.dropout = nn.Dropout(config.drop_rate)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(config.context_length, config.context_length), diagonal=1
            ).bool(),
        )

        # k cache
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        Computes the multi-head attention for the given input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, num_tokens, emb_dim).
            use_cache (bool): If True, cache keys and values for autoregressive generation.

        Returns:
            torch.Tensor: Output tensor of shape (Batch, num_tokens, emb_dim) after attention
                and final projection.
        """
        Batch, num_tokens, d_in = x.shape

        k = self.wk(x)
        q = self.wq(x)
        v = self.wv(x)

        # Split into multiple heads
        q = q.view(Batch, num_tokens, self.n_head, self.head_dim)
        k = k.view(Batch, num_tokens, self.n_head, self.head_dim)
        v = v.view(Batch, num_tokens, self.n_head, self.head_dim)
        # kv cache implementaion
        if use_cache:
            if self.k_cache is None:
                self.k_cache, self.v_cache = k, v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=1)
                self.v_cache = torch.cat([self.v_cache, v], dim=1)
            keys, values = self.k_cache, self.v_cache
        else:
            keys, values = k, v

        # transpose into (B,n_head,T,head_dim)
        q = q.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = q @ keys.transpose(2, 3)

        num_torkens_Q = q.shape[-2]
        num_torkens_K = keys.shape[-2]

        if use_cache:
            mask = self.mask[
                self.ptr_current_pos : self.ptr_current_pos + num_torkens_Q,
                :num_torkens_K,
            ]
            # mask = mask | self.mask[:num_torkens_Q, :num_torkens_K] ## This is for the first token where we need to add 1 row and 1 colunm
            self.ptr_current_pos += num_torkens_Q

        else:
            mask = self.mask[:num_torkens_Q, :num_torkens_K]

        attention_scores = attention_scores.masked_fill_(mask, -torch.inf)
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)
        context = (attention_weights @ values).transpose(1, 2).contiguous()
        context = context.view(Batch, num_tokens, self.d_out)

        return self.out_proj(context)

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.ptr_current_pos = 0

"""
licence : mit
author : amzad hossain rafi
email : amzad.rafi@northsouth.edu

change log :
    18-6-2026 : implement GQA + Sliding Window Attention
                combines GroupedQueryAttention and SlidingWindowAttention
                into a single unified module with SDPA support

references :
    - GQA: https://arxiv.org/abs/2305.13245
    - Sliding Window: Mistral / Mixtral style alternating attention
    - gptoss.py reference implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from BLLMC.components.layers.embeddings import apply_rope, compute_rope_params


class GQASlidingWindowAttention(nn.Module):
    """
    Grouped Query Attention with optional Sliding Window and KV Cache.

    Combines:
        - GQA: fewer KV heads than query heads (n_kv_heads < n_heads)
        - Sliding Window: local attention with bounded KV cache
        - SDPA: fused attention kernel (FlashAttention on GPU)

    Args:
        config: Model config with:
            - emb_dim: embedding dimension
            - n_heads: number of query heads
            - n_kv_heads: number of KV heads (must divide n_heads)
            - context_length: max sequence length
            - drop_rate: dropout probability
            - sliding_window_size: window size (None = full causal)
            - rope_base: RoPE base frequency
            - dtype: data type
    """

    def __init__(self, config):
        super().__init__()
        assert (
            config.emb_dim % config.n_heads == 0
        ), f"emb_dim ({config.emb_dim}) must be divisible by n_heads ({config.n_heads})"
        assert (
            config.n_heads % config.n_kv_heads == 0
        ), f"n_heads ({config.n_heads}) must be divisible by n_kv_heads ({config.n_kv_heads})"

        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads  # GQA repeat factor
        self.dropout_p = config.drop_rate
        self.sliding_window_size = config.sliding_window_size

        # Separate Q and KV projections for GQA
        self.wq = nn.Linear(
            config.emb_dim,
            config.n_heads * self.head_dim,
            bias=False,
            dtype=config.dtype,
        )
        self.wk = nn.Linear(
            config.emb_dim,
            config.n_kv_heads * self.head_dim,
            bias=False,
            dtype=config.dtype,
        )
        self.wv = nn.Linear(
            config.emb_dim,
            config.n_kv_heads * self.head_dim,
            bias=False,
            dtype=config.dtype,
        )
        self.wo = nn.Linear(
            config.emb_dim, config.emb_dim, bias=False, dtype=config.dtype
        )

        # RoPE
        cos, sin = compute_rope_params(
            head_dim=self.head_dim,
            theta_base=getattr(config, "rope_base", 10000.0),
            context_length=config.context_length,
            dtype=config.dtype,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # KV cache
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match query head count via repeat_interleave.

        Args:
            x: (b, n_kv_heads, T, head_dim)

        Returns:
            (b, n_heads, T, head_dim)
        """
        if self.n_rep == 1:
            return x
        return torch.repeat_interleave(x, self.n_rep, dim=1)

    def _update_cache(self, k_new, v_new, num_tokens):
        """
        Append new keys/values to cache, trim by sliding window.

        All tensors in shape (b, n_kv_heads, T, head_dim).

        Returns:
            keys, values for attention (after KV expansion), and absolute
            position of the first key.
        """
        old_len = 0 if self.cache_k is None else self.cache_k.size(2)

        # Concatenate with existing cache
        if self.cache_k is None:
            combined_k, combined_v = k_new, v_new
        else:
            combined_k = torch.cat([self.cache_k, k_new], dim=2)
            combined_v = torch.cat([self.cache_v, v_new], dim=2)

        total_len = combined_k.size(2)

        if self.sliding_window_size is not None:
            W = self.sliding_window_size
            # Keys for attention: W-1 older context + full current chunk
            attn_keep = min(total_len, W + num_tokens - 1)
            keys = combined_k[:, :, -attn_keep:]
            values = combined_v[:, :, -attn_keep:]
            # Cache: only last W tokens (bounded memory)
            cache_keep = min(total_len, W)
            self.cache_k = combined_k[:, :, -cache_keep:]
            self.cache_v = combined_v[:, :, -cache_keep:]
        else:
            keys, values = combined_k, combined_v
            self.cache_k = combined_k
            self.cache_v = combined_v

        # Absolute position of the first retained key
        dropped = total_len - keys.size(2)
        k_start = (self.ptr_current_pos - old_len) + dropped

        return keys, values, k_start

    def _build_mask(self, num_q, num_k, q_start, k_start, device):
        """
        Build boolean attention mask combining causal + sliding window.

        Returns:
            Boolean tensor (num_q, num_k) where True = attend.
        """
        W = (
            self.sliding_window_size
            if self.sliding_window_size is not None
            else num_k + 1
        )
        q_idx = torch.arange(q_start, q_start + num_q, device=device)
        k_idx = torch.arange(k_start, k_start + num_k, device=device)
        diff = q_idx[:, None] - k_idx[None, :]
        return (diff >= 0) & (diff < W)

    def forward(self, x, use_cache=False):
        b, T, _ = x.shape

        # Separate Q, K, V projections (GQA: K/V have fewer heads)
        q = self.wq(x).view(b, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(b, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        q = apply_rope(q, self.cos, self.sin, offset=self.ptr_current_pos)
        k = apply_rope(k, self.cos, self.sin, offset=self.ptr_current_pos)

        if use_cache:
            k, v, k_start = self._update_cache(k, v, T)
            q_start = self.ptr_current_pos
            self.ptr_current_pos += T
        else:
            q_start, k_start = 0, 0
            self.ptr_current_pos = 0

        # Expand KV heads to match Q heads for attention
        k = self._expand_kv(k)
        v = self._expand_kv(v)

        # Attention
        drop = self.dropout_p if self.training else 0.0

        # Fast path: pure causal training with no sliding window → FlashAttention
        if self.sliding_window_size is None and not use_cache:
            context = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=drop
            )
        else:
            mask = self._build_mask(q.size(2), k.size(2), q_start, k_start, q.device)
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=drop
            )

        # Reshape (b, n_heads, T, head_dim) → (b, T, emb_dim) + output projection
        context = context.transpose(1, 2).contiguous().view(b, T, self.config.emb_dim)
        return self.wo(context)

    def reset_cache(self):
        """Clear KV cache between generation runs."""
        self.cache_k = None
        self.cache_v = None
        self.ptr_current_pos = 0

    def __str__(self):
        return (
            f"GQASlidingWindowAttention("
            f"emb_dim={self.config.emb_dim}, "
            f"n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"sliding_window={self.sliding_window_size})"
        )

    def __repr__(self):
        return self.__str__()

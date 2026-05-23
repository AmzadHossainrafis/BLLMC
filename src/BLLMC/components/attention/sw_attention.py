"""
licence : mit
author : amzad hossain rafi
email : amzad.rafi@northsouth.edu

change log :
    23-5-2026 : optimized with fused QKV, SDPA, cleaner cache logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Multi-Head Attention with optional Sliding Window and KV Cache.

    Optimizations over naive implementation:
        - Fused QKV projection (1 matmul instead of 3)
        - PyTorch SDPA for fused attention kernel (FlashAttention on GPU)
        - Separated cache and mask logic into helper methods

    Args:
        d_in: Input embedding dimension.
        d_out: Output dimension (must be divisible by num_heads).
        dropout: Dropout probability on attention weights (training only).
        num_heads: Number of attention heads.
        qkv_bias: Whether Q/K/V projections include bias.
        sliding_window_size: Window size W. None = full causal attention.
    """

    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = config.emb_dim
        self.num_heads = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads
        self.dropout_p = config.drop_rate
        self.sliding_window_size = config.sliding_window_size

        # Fused QKV: single matmul for all three projections
        self.W_qkv = nn.Linear(config.emb_dim, 3 * config.emb_dim, bias=False)
        self.out_proj = nn.Linear(config.emb_dim, config.emb_dim, bias=False)

        # KV cache for autoregressive generation
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def _update_cache(self, k_new, v_new, num_tokens):
        """
        Append new keys/values to cache, trim by sliding window.

        All tensors in shape (b, T, H, D).

        Returns:
            keys, values for attention, and absolute position of the first key.
        """
        old_len = 0 if self.cache_k is None else self.cache_k.size(1)

        # Concatenate with existing cache
        if self.cache_k is None:
            combined_k, combined_v = k_new, v_new
        else:
            combined_k = torch.cat([self.cache_k, k_new], dim=1)
            combined_v = torch.cat([self.cache_v, v_new], dim=1)

        total_len = combined_k.size(1)

        if self.sliding_window_size is not None:
            W = self.sliding_window_size
            # Keys for attention: W-1 older context + full current chunk
            attn_keep = min(total_len, W + num_tokens - 1)
            keys = combined_k[:, -attn_keep:]
            values = combined_v[:, -attn_keep:]
            # Cache: only last W tokens (bounded memory)
            cache_keep = min(total_len, W)
            self.cache_k = combined_k[:, -cache_keep:]
            self.cache_v = combined_v[:, -cache_keep:]
        else:
            keys, values = combined_k, combined_v
            self.cache_k = combined_k
            self.cache_v = combined_v

        # Absolute position of the first retained key
        dropped = total_len - keys.size(1)
        k_start = (self.ptr_current_pos - old_len) + dropped

        return keys, values, k_start

    def _build_mask(self, num_q, num_k, q_start, k_start, device):
        """
        Build boolean attention mask combining causal + sliding window.

        Returns:
            Boolean tensor (num_q, num_k) where True = attend, False = masked.
            (SDPA convention: True means the element takes part in attention)
        """
        W = (
            self.sliding_window_size
            if self.sliding_window_size is not None
            else num_k + 1
        )
        q_idx = torch.arange(q_start, q_start + num_q, device=device)
        k_idx = torch.arange(k_start, k_start + num_k, device=device)
        diff = q_idx[:, None] - k_idx[None, :]  # (Q, K)

        return (diff >= 0) & (diff < W)

    def forward(self, x, use_cache=False):
        b, T, _ = x.shape

        # ── 1. Fused QKV projection ──
        qkv = self.W_qkv(x).view(b, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (b, T, H, D)

        # ── 2. KV cache (operates on b, T, H, D — before transpose) ──
        if use_cache:
            k, v, k_start = self._update_cache(k, v, T)
            q_start = self.ptr_current_pos
            self.ptr_current_pos += T
        else:
            q_start, k_start = 0, 0
            self.ptr_current_pos = 0

        # ── 3. Transpose to (b, H, T, D) for attention ──
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── 4. Attention ──
        drop = self.dropout_p if self.training else 0.0

        # Fast path: pure causal training → enables FlashAttention on GPU
        if self.sliding_window_size is None and not use_cache:
            context = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=drop
            )
        else:
            mask = self._build_mask(q.size(2), k.size(2), q_start, k_start, q.device)
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=drop
            )

        # ── 5. Reshape (b, H, T, D) → (b, T, d_out) + output projection ──
        context = context.transpose(1, 2).contiguous().view(b, T, self.d_out)
        return self.out_proj(context)

    def reset_cache(self):
        """Clear KV cache between generation runs."""
        self.cache_k = None
        self.cache_v = None
        self.ptr_current_pos = 0

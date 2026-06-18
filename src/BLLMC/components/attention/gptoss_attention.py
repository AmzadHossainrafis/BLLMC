# src/BLLMC/components/attention/gptoss_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from BLLMC.components.layers.embeddings import apply_rope, compute_rope_params


class GPTOssAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with optional sliding window, KV Cache,
    and learnable attention sinks.

    Matches the mathematical design of the attention block in notebook/gptoss.py
    while supporting batched inputs.

    Args:
        config: Configuration object.
        sliding_window_size: Window size for local attention (None for full causal).
    """

    def __init__(self, config, sliding_window_size=None):
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
        self.sliding_window_size = sliding_window_size

        # Single fused QKV linear layer
        qkv_dim = self.head_dim * (self.n_heads + 2 * self.n_kv_heads)
        self.qkv = nn.Linear(config.emb_dim, qkv_dim, bias=True, dtype=config.dtype)
        self.wo = nn.Linear(
            config.emb_dim, config.emb_dim, bias=True, dtype=config.dtype
        )

        # Learnable attention sinks
        self.sinks = nn.Parameter(torch.zeros(self.n_heads, dtype=config.dtype))

        # Rotary Position Embeddings (RoPE)
        cos, sin = compute_rope_params(
            head_dim=self.head_dim,
            theta_base=getattr(config, "rope_base", 150000.0),
            context_length=config.context_length,
            dtype=config.dtype,
            scaling_factor=getattr(config, "rope_scaling_factor", 1.0),
            initial_context_length=getattr(config, "context_length", 4096),
            ntk_alpha=getattr(config, "rope_ntk_alpha", 1.0),
            ntk_beta=getattr(config, "rope_ntk_beta", 32.0),
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # KV cache
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match query group shape.

        Args:
            x: (b, n_kv_heads, T, head_dim)

        Returns:
            (b, n_kv_heads, n_rep, T, head_dim)
        """
        return x[:, :, None, :, :].expand(-1, -1, self.n_rep, -1, -1)

    def _update_cache(self, k_new, v_new, num_tokens):
        """
        Append new keys/values to cache, trim by sliding window.

        All tensors in shape (b, n_kv_heads, T, head_dim).

        Returns:
            keys, values for attention, and absolute position of the first key.
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

        # 1. Project to fused QKV
        qkv = self.qkv(x)
        q = qkv[:, :, : self.n_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            :,
            self.n_heads
            * self.head_dim : (self.n_heads + self.n_kv_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            :,
            (self.n_heads + self.n_kv_heads) * self.head_dim :,
        ].contiguous()

        # 2. Reshape to split heads
        q = q.view(b, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE
        q = apply_rope(q, self.cos, self.sin, offset=self.ptr_current_pos)
        k = apply_rope(k, self.cos, self.sin, offset=self.ptr_current_pos)

        # 4. Update and fetch cache if requested
        if use_cache:
            k, v, k_start = self._update_cache(k, v, T)
            q_start = self.ptr_current_pos
            self.ptr_current_pos += T
        else:
            q_start, k_start = 0, 0
            self.ptr_current_pos = 0

        # 5. Expand KV heads for GQA compatibility
        k_exp = self._expand_kv(k)
        v_exp = self._expand_kv(v)

        # Reshape Q to group attention form (b, n_kv_heads, n_rep, T, head_dim)
        q_gqa = q.view(b, self.n_kv_heads, self.n_rep, T, self.head_dim)

        # 6. Attention scoring with sinks
        sm_scale = 1.0 / math.sqrt(self.head_dim)
        # scores shape: (b, n_kv_heads, n_rep, T, T_k)
        scores = torch.einsum("bhmqd,bhmkd->bhmqk", q_gqa, k_exp) * sm_scale

        # Apply causal/sliding window mask
        mask = self._build_mask(T, k.size(2), q_start, k_start, x.device)
        scores = scores.masked_fill(~mask, float("-inf"))

        # Concatenate learnable attention sinks along key dimension
        # sinks shape: (n_heads,) -> view as (1, n_kv_heads, n_rep, 1, 1) and expand
        sinks_expanded = self.sinks.view(1, self.n_kv_heads, self.n_rep, 1, 1).expand(
            b, -1, -1, T, 1
        )
        scores = torch.cat([scores, sinks_expanded], dim=-1)

        # Compute softmax over expanded keys + sink column
        weights = torch.softmax(scores, dim=-1)

        # Drop the sink column to keep only the text token weights
        weights = weights[..., :-1]

        # Apply dropout to attention weights (training only)
        if self.training and self.dropout_p > 0.0:
            weights = F.dropout(weights, p=self.dropout_p)

        # 7. Compute context output
        # weights: (b, h, m, q, k), v_exp: (b, h, m, k, d)
        attn = torch.einsum("bhmqk,bhmkd->bhmqd", weights, v_exp)

        # 8. Permute and flatten heads back to embedding dimension
        attn = attn.permute(0, 3, 1, 2, 4).reshape(b, T, -1)

        # 9. Output projection
        return self.wo(attn)

    def reset_cache(self):
        """Clear KV cache between generation runs."""
        self.cache_k = None
        self.cache_v = None
        self.ptr_current_pos = 0

    def __str__(self):
        return (
            f"GPTOssAttention("
            f"emb_dim={self.config.emb_dim}, "
            f"n_heads={self.n_heads}, "
            f"n_kv_heads={self.n_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"sliding_window={self.sliding_window_size})"
        )

    def __repr__(self):
        return self.__str__()

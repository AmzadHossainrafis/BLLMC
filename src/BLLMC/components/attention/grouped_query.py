"""
Author: amzad hossain rafi
email : amzad.rafi@northsouth.edu
licence : mit

change log :
    1-6-2026 : start
    1-6-2026 : implement grouped query attention with per head kv cache


"""

import torch
import torch.nn as nn
from BLLMC.components.layers.embeddings import apply_rope
from BLLMC.components.layers.embeddings import compute_rope_params


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_heads == 0
        assert config.n_heads % config.n_key_val_groups == 0
        self.config = config
        self.head_dim = config.emb_dim // config.n_heads

        self.wk = nn.Linear(
            config.emb_dim, config.n_key_val_groups * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.emb_dim, config.n_key_val_groups * self.head_dim, bias=False
        )

        self.wq = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.wo = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.dropout = nn.Dropout(config.drop_rate)

        # k cache
        cos, sin = compute_rope_params(
            head_dim=self.head_dim,
            context_length=config.context_length,
            theta_base=config.rope_base,
            dtype=config.dtype,
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.k_cache = None
        self.v_cache = None
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor, use_cache=False):
        b, t, c = x.shape

        # Projections
        # Shape: (b, t, n_groups, head_dim) -> transpose to (b, n_groups, t, head_dim)
        k = (
            self.wk(x)
            .view(b, t, self.config.n_key_val_groups, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.wv(x)
            .view(b, t, self.config.n_key_val_groups, self.head_dim)
            .transpose(1, 2)
        )
        q = self.wq(x).view(b, t, self.config.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Query and Key
        q = apply_rope(q, self.cos, self.sin, offset=self.ptr_current_pos)
        k = apply_rope(k, self.cos, self.sin, offset=self.ptr_current_pos)

        if use_cache:
            if self.k_cache is None:
                self.k_cache = k
                self.v_cache = v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
            key_base, val_base = self.k_cache, self.v_cache
        else:
            key_base, val_base = k, v
            self.k_cache, self.v_cache = None, None
            self.ptr_current_pos = 0

        # Repeat key/value groups to match the number of query heads
        repeats = self.config.n_heads // self.config.n_key_val_groups
        k_head = torch.repeat_interleave(key_base, repeats, dim=1)
        v_head = torch.repeat_interleave(val_base, repeats, dim=1)

        # Matmul: (b, n_heads, t, head_dim) @ (b, n_heads, head_dim, seq_len_K) -> (b, n_heads, t, seq_len_K)
        score = q @ k_head.transpose(-2, -1)

        num_tokens_Q = q.shape[-2]
        num_tokens_K = k_head.shape[-2]

        # Dynamic causal mask based on absolute position tracking
        if use_cache:
            q_positions = torch.arange(
                self.ptr_current_pos,
                self.ptr_current_pos + num_tokens_Q,
                device=q.device,
                dtype=torch.long,
            )
            self.ptr_current_pos += num_tokens_Q
        else:
            q_positions = torch.arange(num_tokens_Q, device=q.device, dtype=torch.long)
            self.ptr_current_pos = 0

        k_positions = torch.arange(num_tokens_K, device=k.device, dtype=torch.long)

        # Mask positions where query index < key index
        q_mask = q_positions[:, None] < k_positions[None, :]
        atten_score = score.masked_fill_(q_mask, -torch.inf)

        # Softmax & dropout
        attention_weights = torch.softmax(
            atten_score / (k_head.shape[-1] ** 0.5), dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        # Context computation
        context = (attention_weights @ v_head).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens_Q, self.config.emb_dim)

        return self.wo(context)

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.ptr_current_pos = 0

    def __str__(self):
        return (
            f"GroupedQueryAttention("
            f"emb_dim={self.config.emb_dim}, "
            f"n_heads={self.config.n_heads}, "
            f"n_key_val_groups={self.config.n_key_val_groups}, "
            f"head_dim={self.head_dim})"
        )

    def __repr__(self):
        return self.__str__()

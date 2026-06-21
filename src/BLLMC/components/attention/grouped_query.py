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
        assert config.n_heads % config.n_kv_heads == 0
        self.config = config
        self.head_dim = config.emb_dim // config.n_heads

        self.wk = nn.Linear(
            config.emb_dim, config.n_kv_heads * self.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.emb_dim, config.n_kv_heads * self.head_dim, bias=False
        )

        self.wq = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
        self.wo = nn.Linear(config.emb_dim, config.emb_dim, bias=False)

        cos, sin = compute_rope_params(
            head_dim=self.head_dim,
            context_length=config.context_length,
            theta_base=config.rope_base,
            dtype=config.dtype,
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x: torch.Tensor, use_cache=False):
        b, t, c = x.shape
        k = self.wk(x).view(b, t, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(b, t, self.config.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.wq(x).view(b, t, self.config.n_heads, self.head_dim).transpose(1, 2)
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

        repeats = self.config.n_heads // self.config.n_kv_heads
        k_head = torch.repeat_interleave(key_base, repeats, dim=1)
        v_head = torch.repeat_interleave(val_base, repeats, dim=1)

        score = q @ k_head.transpose(-2, -1)

        num_tokens_Q = q.shape[-2]
        num_tokens_K = k_head.shape[-2]
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

        q_mask = q_positions[:, None] < k_positions[None, :]
        attention_score = score / (k_head.shape[-1] ** 0.5)
        attention_score = attention_score.masked_fill_(q_mask, -torch.inf)

        attention_weights = torch.softmax(attention_score, dim=-1)

        context = (attention_weights @ v_head).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens_Q, self.config.emb_dim)

        return self.wo(context)

    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None
        self.ptr_current_pos = 0

    def __str__(self):
        return (
            f"GroupedQueryAttention("
            f"emb_dim={self.config.emb_dim}, "
            f"n_heads={self.config.n_heads}, "
            f"n_kv_heads={self.config.n_kv_heads}, "
            f"head_dim={self.head_dim})"
        )

    def __repr__(self):
        return self.__str__()

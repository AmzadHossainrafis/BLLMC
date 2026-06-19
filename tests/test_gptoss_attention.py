# tests/test_gptoss_attention.py
import math
from dataclasses import dataclass
import pytest
import torch
import torch.nn as nn

from BLLMC.components.attention.gptoss_attention import GPTOssAttention
from BLLMC.components.blocks.gptoss_block import GPTOssBlock
from BLLMC.components.layers.embeddings import apply_rope


@dataclass
class GPTOssTestConfig:
    emb_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 2
    context_length: int = 64
    drop_rate: float = 0.0
    rope_base: float = 10000.0
    sliding_window_size: int = 16
    dtype: object = None
    num_experts: int = 4
    num_experts_per_tok: int = 1
    moe_hidden_dim: int = 128

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.float32


@pytest.fixture
def config():
    return GPTOssTestConfig()


@pytest.fixture
def attn_model(config):
    torch.manual_seed(42)
    m = GPTOssAttention(config, sliding_window_size=config.sliding_window_size)
    m.eval()
    return m


# ────────────────────────────────────────────────────────────────
# 1. Forward Pass Shape & Health Tests
# ────────────────────────────────────────────────────────────────


class TestGPTOssForwardPass:
    def test_output_shape(self, attn_model):
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = attn_model(x)
        assert out.shape == x.shape

    def test_no_nan_or_inf(self, attn_model):
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = attn_model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ────────────────────────────────────────────────────────────────
# 2. GQA Expansion Tests
# ────────────────────────────────────────────────────────────────


class TestGPTOssGQAExpansion:
    def test_expansion_correctness(self, attn_model):
        kv = torch.randn(2, 2, 8, 16)  # (b, n_kv_heads=2, T=8, head_dim=16)
        expanded = attn_model._expand_kv(kv)
        assert expanded.shape == (2, 2, 4, 8, 16)  # (b, n_kv_heads, n_rep, T, head_dim)

        # Verify repetition: all heads within a group must be identical
        for group in range(2):
            for head in range(4):
                assert torch.equal(expanded[:, group, head, :, :], kv[:, group, :, :])


# ────────────────────────────────────────────────────────────────
# 3. Attention Sinks Tests
# ────────────────────────────────────────────────────────────────


class TestGPTOssAttentionSinks:
    def test_sinks_pressure_valve(self, config):
        """
        Verify that having a higher sink value results in smaller
        attention weights allocated to the actual sequence tokens.
        """
        m = GPTOssAttention(config, sliding_window_size=None)
        m.eval()

        x = torch.randn(1, 8, 128)

        # Run with default sinks (initialized to 0)
        with torch.no_grad():
            # Let's extract weights by replicating forward logic
            qkv = m.qkv(x)
            q = (
                qkv[:, :, : m.n_heads * m.head_dim]
                .view(1, 8, m.n_heads, m.head_dim)
                .transpose(1, 2)
            )
            k = (
                qkv[
                    :,
                    :,
                    m.n_heads * m.head_dim : (m.n_heads + m.n_kv_heads) * m.head_dim,
                ]
                .view(1, 8, m.n_kv_heads, m.head_dim)
                .transpose(1, 2)
            )

            q = apply_rope(q, m.cos, m.sin, offset=0)
            k = apply_rope(k, m.cos, m.sin, offset=0)

            k_exp = m._expand_kv(k)
            q_gqa = q.view(1, m.n_kv_heads, m.n_rep, 8, m.head_dim)

            scores = torch.einsum("bhmqd,bhmkd->bhmqk", q_gqa, k_exp) / math.sqrt(
                m.head_dim
            )
            mask = m._build_mask(8, 8, 0, 0, x.device)
            scores = scores.masked_fill(~mask, float("-inf"))

            # Scenario A: sinks are zero
            sinks_zero = torch.zeros(1, m.n_kv_heads, m.n_rep, 8, 1)
            scores_a = torch.cat([scores, sinks_zero], dim=-1)
            weights_a = torch.softmax(scores_a, dim=-1)[..., :-1]
            sum_a = weights_a.sum(dim=-1)  # Sum over text tokens

            # Scenario B: sinks are very large (high pressure valve value)
            sinks_large = torch.full((1, m.n_kv_heads, m.n_rep, 8, 1), 10.0)
            scores_b = torch.cat([scores, sinks_large], dim=-1)
            weights_b = torch.softmax(scores_b, dim=-1)[..., :-1]
            sum_b = weights_b.sum(dim=-1)

            # Sum of B weights should be strictly smaller because sink absorbed attention
            assert torch.all(sum_b < sum_a)


# ────────────────────────────────────────────────────────────────
# 4. KV Cache Parity Tests
# ────────────────────────────────────────────────────────────────


class TestGPTOssKVCache:
    def test_cache_matches_full_pass(self, config):
        m = GPTOssAttention(config, sliding_window_size=8)
        m.eval()
        x = torch.randn(1, 8, 128)

        # Full forward pass (no cache)
        with torch.no_grad():
            out_full = m(x, use_cache=False)

        # Cache prefill + decode pass
        m.reset_cache()
        with torch.no_grad():
            out_prefill = m(x[:, :4, :], use_cache=True)
            outs_decode = []
            for i in range(4, 8):
                out_i = m(x[:, i : i + 1, :], use_cache=True)
                outs_decode.append(out_i)

        out_incremental = torch.cat([out_prefill] + outs_decode, dim=1)
        diff = (out_full - out_incremental).abs().max().item()
        assert diff < 1e-4, f"Cache output mismatch: {diff}"

    def test_cache_reset(self, attn_model):
        x = torch.randn(1, 4, 128)
        attn_model(x, use_cache=True)
        assert attn_model.cache_k is not None
        attn_model.reset_cache()
        assert attn_model.cache_k is None
        assert attn_model.cache_v is None
        assert attn_model.ptr_current_pos == 0


# ────────────────────────────────────────────────────────────────
# 5. Gradient Flow Tests
# ────────────────────────────────────────────────────────────────


class TestGPTOssGradientFlow:
    def test_gradients(self, config):
        m = GPTOssAttention(config, sliding_window_size=8)
        m.train()
        x = torch.randn(2, 6, 128, requires_grad=True)
        out = m(x)
        out.sum().backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert m.sinks.grad is not None
        assert not torch.isnan(m.sinks.grad).any()


# ────────────────────────────────────────────────────────────────
# 6. GPTOssBlock Tests
# ────────────────────────────────────────────────────────────────


class TestGPTOssBlock:
    def test_block_alternating_sliding_window(self, config):
        # Even layer index: should have sliding window size
        block_even = GPTOssBlock(config, layer_idx=0)
        assert block_even.attn.sliding_window_size == config.sliding_window_size

        # Odd layer index: should have NO sliding window (None)
        block_odd = GPTOssBlock(config, layer_idx=1)
        assert block_odd.attn.sliding_window_size is None

    def test_block_forward_shapes(self, config):
        block = GPTOssBlock(config, layer_idx=0)
        block.eval()
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape

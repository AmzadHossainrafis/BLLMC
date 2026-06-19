"""
Pytest test suite for GQASlidingWindowAttention.

Verifies correctness against gptoss.py's AttentionBlock implementation
including shape, GQA expansion, sliding window masking, KV cache,
gradient flow, and cross-implementation consistency.
"""

import math
from dataclasses import dataclass

import pytest
import torch

from BLLMC.components.attention.gqa_sliding_window import GQASlidingWindowAttention
from BLLMC.components.layers.embeddings import apply_rope, compute_rope_params

# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────


@dataclass
class GQATestConfig:
    emb_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 2
    context_length: int = 64
    drop_rate: float = 0.0
    rope_base: float = 10000.0
    sliding_window_size: int = 16
    dtype: object = None

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.float32


@pytest.fixture
def config():
    return GQATestConfig()


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    m = GQASlidingWindowAttention(config)
    m.eval()
    return m


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


class TestForwardPass:
    """Basic forward pass shape and value checks."""

    def test_output_shape(self, model):
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_no_nan(self, model):
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_no_inf(self, model):
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = model(x)
        assert not torch.isinf(out).any()


class TestGQAExpansion:
    """Verify KV head expansion for grouped query attention."""

    def test_expansion_shape(self, model):
        kv = torch.randn(2, 2, 8, 16)  # (b, n_kv_heads=2, T=8, head_dim=16)
        expanded = model._expand_kv(kv)
        assert expanded.shape == (2, 8, 8, 16)

    def test_expansion_correctness(self, model):
        kv = torch.randn(2, 2, 8, 16)
        expanded = model._expand_kv(kv)
        # n_rep = 8 // 2 = 4, so heads 0-3 should equal kv_head 0
        for i in range(4):
            assert torch.equal(expanded[:, i, :, :], kv[:, 0, :, :])
        # heads 4-7 should equal kv_head 1
        for i in range(4, 8):
            assert torch.equal(expanded[:, i, :, :], kv[:, 1, :, :])

    def test_no_expansion_when_ratio_is_one(self):
        cfg = GQATestConfig(n_heads=8, n_kv_heads=8)
        m = GQASlidingWindowAttention(cfg)
        kv = torch.randn(1, 8, 4, 16)
        result = m._expand_kv(kv)
        assert torch.equal(result, kv), "Should return input unchanged when n_rep=1"


class TestSlidingWindowMask:
    """Verify sliding window attention mask patterns."""

    def test_full_causal_when_seq_smaller_than_window(self, model):
        """When seq_len < window_size, mask should be purely causal."""
        mask = model._build_mask(8, 8, 0, 0, torch.device("cpu"))
        expected = torch.tril(torch.ones(8, 8, dtype=torch.bool))
        assert torch.equal(mask, expected)

    def test_window_limits_attention_span(self):
        """Each row should have at most `window_size` visible positions."""
        cfg = GQATestConfig(sliding_window_size=3)
        m = GQASlidingWindowAttention(cfg)
        mask = m._build_mask(8, 8, 0, 0, torch.device("cpu"))
        for i in range(8):
            assert mask[i].sum().item() <= 3

    def test_window_3_pattern(self):
        """Verify exact mask pattern for window=3."""
        cfg = GQATestConfig(sliding_window_size=3)
        m = GQASlidingWindowAttention(cfg)
        mask = m._build_mask(8, 8, 0, 0, torch.device("cpu"))

        # Row i should attend to positions max(0, i-2) through i
        for i in range(8):
            for j in range(8):
                expected = (j <= i) and (i - j < 3)
                assert (
                    mask[i, j].item() == expected
                ), f"Mismatch at ({i},{j}): got {mask[i,j].item()}, expected {expected}"

    def test_matches_gptoss_mask_pattern(self):
        """Verify our mask matches the gptoss.py triu+tril construction."""
        seq_len = 8
        window = 4
        cfg = GQATestConfig(sliding_window_size=window)
        m = GQASlidingWindowAttention(cfg)

        # gptoss builds: triu(-inf, diagonal=1) + tril(-inf, diagonal=-window)
        gptoss_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=1
        )
        gptoss_mask += torch.tril(
            torch.full((seq_len, seq_len), float("-inf")), diagonal=-window
        )
        gptoss_visible = gptoss_mask == 0

        our_mask = m._build_mask(seq_len, seq_len, 0, 0, torch.device("cpu"))
        assert torch.equal(gptoss_visible, our_mask)


class TestFullCausalAttention:
    """Test full causal (no sliding window) path."""

    def test_forward_no_window(self):
        cfg = GQATestConfig(sliding_window_size=None)
        m = GQASlidingWindowAttention(cfg)
        m.eval()
        x = torch.randn(2, 16, 128)
        with torch.no_grad():
            out = m(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


class TestKVCache:
    """Verify KV cache produces consistent results with full forward pass."""

    def test_cache_matches_full_pass(self):
        cfg = GQATestConfig(sliding_window_size=16)
        m = GQASlidingWindowAttention(cfg)
        m.eval()
        x = torch.randn(1, 8, 128)

        # Full forward pass
        with torch.no_grad():
            out_full = m(x, use_cache=False)

        # Incremental: prefill + token-by-token decode
        m.reset_cache()
        with torch.no_grad():
            out_prefill = m(x[:, :4, :], use_cache=True)
            outs_decode = []
            for i in range(4, 8):
                out_i = m(x[:, i : i + 1, :], use_cache=True)
                outs_decode.append(out_i)

        out_incremental = torch.cat([out_prefill] + outs_decode, dim=1)
        diff = (out_full - out_incremental).abs().max().item()
        assert diff < 1e-4, f"Cache inference too different: {diff}"

    def test_cache_matches_full_pass_no_window(self):
        cfg = GQATestConfig(sliding_window_size=None)
        m = GQASlidingWindowAttention(cfg)
        m.eval()
        x = torch.randn(1, 8, 128)

        with torch.no_grad():
            out_full = m(x, use_cache=False)

        m.reset_cache()
        with torch.no_grad():
            out_prefill = m(x[:, :4, :], use_cache=True)
            outs_decode = []
            for i in range(4, 8):
                out_i = m(x[:, i : i + 1, :], use_cache=True)
                outs_decode.append(out_i)

        out_incremental = torch.cat([out_prefill] + outs_decode, dim=1)
        diff = (out_full - out_incremental).abs().max().item()
        assert diff < 1e-4, f"Cache inference too different: {diff}"

    def test_reset_cache_clears_state(self):
        cfg = GQATestConfig()
        m = GQASlidingWindowAttention(cfg)
        m.eval()
        x = torch.randn(1, 4, 128)

        with torch.no_grad():
            m(x, use_cache=True)

        assert m.cache_k is not None
        m.reset_cache()
        assert m.cache_k is None
        assert m.cache_v is None
        assert m.ptr_current_pos == 0


class TestGradientFlow:
    """Verify gradients propagate correctly."""

    def test_input_gradient(self):
        cfg = GQATestConfig(sliding_window_size=8)
        m = GQASlidingWindowAttention(cfg)
        m.train()

        x = torch.randn(2, 10, 128, requires_grad=True)
        out = m(x)
        out.sum().backward()

        assert x.grad is not None, "No gradient on input"
        assert not torch.isnan(x.grad).any(), "NaN in input gradient"

    def test_parameter_gradients(self):
        cfg = GQATestConfig(sliding_window_size=8)
        m = GQASlidingWindowAttention(cfg)
        m.train()

        x = torch.randn(2, 10, 128)
        out = m(x)
        out.sum().backward()

        for name, p in m.named_parameters():
            if p.grad is not None:
                assert not torch.isnan(p.grad).any(), f"NaN grad in: {name}"


class TestCrossVerifyGptoss:
    """Cross-verify attention computation against gptoss.py's approach."""

    def test_manual_attention_matches_sdpa(self):
        """Manual QKV matmul with mask should match SDPA output."""
        torch.manual_seed(123)
        seq_len, emb_dim, n_heads, n_kv_heads = 8, 128, 8, 2
        head_dim = emb_dim // n_heads
        window = 4

        cfg = GQATestConfig(
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            sliding_window_size=window,
            context_length=64,
        )
        m = GQASlidingWindowAttention(cfg)
        m.eval()

        x = torch.randn(1, seq_len, emb_dim)

        with torch.no_grad():
            # Manual attention
            q = m.wq(x).view(1, seq_len, n_heads, head_dim).transpose(1, 2)
            k = m.wk(x).view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)
            v = m.wv(x).view(1, seq_len, n_kv_heads, head_dim).transpose(1, 2)

            q = apply_rope(q, m.cos, m.sin, offset=0)
            k = apply_rope(k, m.cos, m.sin, offset=0)

            k_exp = m._expand_kv(k)
            v_exp = m._expand_kv(v)

            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) * scale

            mask = m._build_mask(seq_len, seq_len, 0, 0, x.device)
            scores = scores.masked_fill(~mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)
            manual_out = torch.matmul(attn_weights, v_exp)
            manual_out = (
                manual_out.transpose(1, 2).contiguous().view(1, seq_len, emb_dim)
            )
            manual_out = m.wo(manual_out)

            # SDPA path
            sdpa_out = m(x)

        diff = (manual_out - sdpa_out).abs().max().item()
        assert diff < 1e-5, f"Manual vs SDPA mismatch: {diff}"

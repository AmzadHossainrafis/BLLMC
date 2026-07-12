"""
Unit tests for Rotary Position Embeddings (RoPE).

Tests cover:
    - Output shapes of compute_rope_params
    - Shape preservation in apply_rope
    - Offset (KV cache) support
    - Numerical properties (unit norm rotations)
    - Edge cases (single token, max context)
"""

import math
import pytest
import torch
from BLLMC.components.layers.embeddings import compute_rope_params, apply_rope

# ─── compute_rope_params Shape Tests ──────────────────────────────


class TestComputeRopeParamsShape:
    """Test that precomputed cos/sin tables have correct shapes."""

    def test_basic_shape(self):
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

    def test_small_head_dim(self):
        cos, sin = compute_rope_params(head_dim=8, context_length=32)
        assert cos.shape == (32, 8)
        assert sin.shape == (32, 8)

    def test_large_context(self):
        cos, sin = compute_rope_params(head_dim=64, context_length=8192)
        assert cos.shape == (8192, 64)
        assert sin.shape == (8192, 64)

    def test_head_dim_128(self):
        """LLaMA 3 uses head_dim=128 (4096/32)."""
        cos, sin = compute_rope_params(head_dim=128, context_length=4096)
        assert cos.shape == (4096, 128)
        assert sin.shape == (4096, 128)

    def test_odd_head_dim_raises(self):
        """Head dim must be even for RoPE to work."""
        with pytest.raises(AssertionError):
            compute_rope_params(head_dim=7, context_length=32)


# ─── compute_rope_params Numerical Properties ────────────────────


class TestComputeRopeParamsNumerical:
    """Test mathematical properties of RoPE tables."""

    def test_cos_sin_bounded(self):
        """cos and sin values must be in [-1, 1]."""
        cos, sin = compute_rope_params(head_dim=64, context_length=256)
        assert cos.min() >= -1.0
        assert cos.max() <= 1.0
        assert sin.min() >= -1.0
        assert sin.max() <= 1.0

    def test_cos_sin_unit_circle(self):
        """cos² + sin² ≈ 1 for all positions and dimensions."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        identity = cos**2 + sin**2
        assert torch.allclose(identity, torch.ones_like(identity), atol=1e-5)

    def test_position_zero_cos_is_one(self):
        """At position 0, cos(0) = 1 for all frequencies."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        assert torch.allclose(cos[0], torch.ones(64), atol=1e-5)

    def test_position_zero_sin_is_zero(self):
        """At position 0, sin(0) = 0 for all frequencies."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        assert torch.allclose(sin[0], torch.zeros(64), atol=1e-5)

    def test_different_theta_base(self):
        """Higher theta_base should produce lower-frequency rotations."""
        cos_low, _ = compute_rope_params(
            head_dim=64, context_length=128, theta_base=10_000
        )
        cos_high, _ = compute_rope_params(
            head_dim=64, context_length=128, theta_base=500_000
        )
        # With higher base, cos values at same position should be closer to 1
        # (slower rotation), especially for high-frequency dimensions
        assert cos_high[10].mean() > cos_low[10].mean()

    def test_default_dtype_is_float32(self):
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32


# ─── apply_rope Shape Tests ───────────────────────────────────────


class TestApplyRopeShape:
    """Test that apply_rope preserves input shapes."""

    def test_basic_shape(self):
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(2, 8, 32, 64)  # (batch, heads, seq_len, head_dim)
        out = apply_rope(x, cos, sin)
        assert out.shape == (2, 8, 32, 64)

    def test_single_head(self):
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 1, 16, 64)
        out = apply_rope(x, cos, sin)
        assert out.shape == (1, 1, 16, 64)

    def test_single_token(self):
        """Single token forward (inference mode)."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 1, 64)
        out = apply_rope(x, cos, sin)
        assert out.shape == (1, 4, 1, 64)

    def test_large_batch(self):
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(16, 8, 32, 64)
        out = apply_rope(x, cos, sin)
        assert out.shape == (16, 8, 32, 64)

    def test_full_context_length(self):
        """Sequence length equals max context length."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 128, 64)
        out = apply_rope(x, cos, sin)
        assert out.shape == (1, 4, 128, 64)


# ─── apply_rope with Offset (KV Cache) ───────────────────────────


class TestApplyRopeOffset:
    """Test that offset parameter works correctly for cached generation."""

    def test_offset_preserves_shape(self):
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 1, 64)
        out = apply_rope(x, cos, sin, offset=10)
        assert out.shape == (1, 4, 1, 64)

    def test_offset_gives_different_result(self):
        """Same input at different positions should produce different outputs."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 1, 64)
        out_pos0 = apply_rope(x, cos, sin, offset=0)
        out_pos5 = apply_rope(x, cos, sin, offset=5)
        assert not torch.allclose(out_pos0, out_pos5)

    def test_offset_matches_full_sequence(self):
        """Token at offset=t should match position t in a full-sequence forward."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 5, 64)

        # Full sequence
        full_out = apply_rope(x, cos, sin, offset=0)

        # Token-by-token with offset
        for t in range(5):
            token = x[:, :, t : t + 1, :]
            token_out = apply_rope(token, cos, sin, offset=t)
            assert torch.allclose(
                full_out[:, :, t : t + 1, :], token_out, atol=1e-5
            ), f"Mismatch at position {t}"

    def test_max_offset(self):
        """Offset near end of context window should still work."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 1, 64)
        out = apply_rope(x, cos, sin, offset=127)
        assert out.shape == (1, 4, 1, 64)


# ─── apply_rope Numerical Properties ─────────────────────────────


class TestApplyRopeNumerical:
    """Test mathematical properties of the rotation."""

    def test_preserves_dtype(self):
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 8, 64)
        out = apply_rope(x, cos, sin)
        assert out.dtype == x.dtype

    def test_preserves_norm(self):
        """RoPE is a rotation — it should preserve vector norms."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 8, 64)
        out = apply_rope(x, cos, sin)
        x_norm = torch.norm(x, dim=-1)
        out_norm = torch.norm(out, dim=-1)
        assert torch.allclose(
            x_norm, out_norm, atol=1e-4
        ), "RoPE should preserve vector norms (it's a rotation)"

    def test_position_zero_is_identity(self):
        """At position 0, cos=1 and sin=0, so output should equal input."""
        cos, sin = compute_rope_params(head_dim=64, context_length=128)
        x = torch.randn(1, 4, 1, 64)
        out = apply_rope(x, cos, sin, offset=0)
        assert torch.allclose(
            x, out, atol=1e-5
        ), "RoPE at position 0 should be identity"

    def test_yarn_scaling_preserves_shape_and_adjusts_values(self):
        """Test that YaRN scaling computes tables of correct shape and adjusts frequency scaling."""
        cos, sin = compute_rope_params(
            head_dim=64, context_length=128, scaling_factor=2.0
        )
        assert cos.shape == (128, 64)
        assert sin.shape == (128, 64)

        # Norms should be scaled by the concentration factor (1.0 + 0.1 * ln(2))
        expected_concentration = 0.1 * math.log(2.0) + 1.0
        identity = cos**2 + sin**2
        expected_identity = torch.full_like(identity, expected_concentration**2)
        assert torch.allclose(identity, expected_identity, atol=1e-5)

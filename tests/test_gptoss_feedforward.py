# tests/test_gptoss_feedforward.py
from dataclasses import dataclass
import pytest
import torch
import torch.nn as nn

from BLLMC.components.layers.activations import swiglu
from BLLMC.components.layers.feedforward import GPTOssFeedForward
from BLLMC.components.blocks.gptoss_block import GPTOssBlock


@dataclass
class GPTOssFeedForwardTestConfig:
    emb_dim: int = 128
    n_heads: int = 8
    n_kv_heads: int = 2
    context_length: int = 64
    drop_rate: float = 0.0
    rope_base: float = 10000.0
    sliding_window_size: int = 16
    dtype: object = None
    num_experts: int = 4
    num_experts_per_tok: int = 2
    moe_hidden_dim: int = 256
    swiglu_limit: float = 7.0

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.float32


@pytest.fixture
def config():
    return GPTOssFeedForwardTestConfig()


@pytest.fixture
def moe_model(config):
    torch.manual_seed(42)
    m = GPTOssFeedForward(config)
    m.eval()
    return m


# ────────────────────────────────────────────────────────────────
# 1. SwiGLU Activation Tests
# ────────────────────────────────────────────────────────────────

class TestSwiGLUActivation:
    def test_swiglu_shape_reduction(self):
        x = torch.randn(2, 4, 64)
        out = swiglu(x)
        # Splicing in half reduces last dim by 2
        assert out.shape == (2, 4, 32)

    def test_swiglu_clamp_limits(self):
        # Create inputs larger than the limit (7.0) and smaller than -7.0
        x = torch.zeros(1, 1, 4)
        x[0, 0, 0] = 50.0   # glu part, index 0 (even)
        x[0, 0, 1] = 50.0   # linear part, index 1 (odd)
        x[0, 0, 2] = -50.0  # glu part, index 2 (even)
        x[0, 0, 3] = -50.0  # linear part, index 3 (odd)

        # Let's run swiglu with limit=7.0, alpha=1.702
        out = swiglu(x, alpha=1.702, limit=7.0)

        # Expected even values (glu):
        # idx 0: clamped to 7.0 -> out_glu_0 = 7.0 * sigmoid(1.702 * 7.0) ≈ 7.0
        # idx 2: -50.0 (no min clamping in reference for glu branch) -> out_glu_2 = -50.0 * sigmoid(1.702 * -50.0) ≈ 0.0

        # Expected odd values (linear):
        # idx 1: clamped to 7.0 -> out_linear_1 = 7.0
        # idx 3: clamped to -7.0 -> out_linear_3 = -7.0

        # Output shape is (1, 1, 2).
        # Out[0,0,0] corresponds to even=idx 0 and odd=idx 1:
        # Out[0,0,0] = out_glu_0 * (out_linear_1 + 1.0) ≈ 7.0 * (7.0 + 1.0) = 56.0
        # Out[0,0,1] corresponds to even=idx 2 and odd=idx 3:
        # Out[0,0,1] = out_glu_2 * (out_linear_3 + 1.0) ≈ 0.0 * (-7.0 + 1.0) = 0.0

        assert torch.allclose(out[0, 0, 0], torch.tensor(56.0), atol=1e-1)
        assert torch.allclose(out[0, 0, 1], torch.tensor(0.0), atol=1e-2)


# ────────────────────────────────────────────────────────────────
# 2. GPTOssFeedForward Shape & Health Tests
# ────────────────────────────────────────────────────────────────

class TestGPTOssFeedForwardShapes:
    def test_output_shape(self, moe_model):
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out = moe_model(x)
        assert out.shape == x.shape

    def test_no_nan_or_inf(self, moe_model):
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out = moe_model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ────────────────────────────────────────────────────────────────
# 3. Routing & Expert Gate Tests
# ────────────────────────────────────────────────────────────────

class TestGPTOssRouting:
    def test_routing_correctness(self, config):
        m = GPTOssFeedForward(config)
        m.eval()
        
        x = torch.randn(1, 1, 128)
        # Set gate weights such that token is strongly routed to expert 0 and 2
        with torch.no_grad():
            m.gate.weight.zero_()
            m.gate.bias.zero_()
            m.gate.bias[0] = 100.0  # Expert 0
            m.gate.bias[2] = 50.0   # Expert 2
            
            # Forward pass
            out = m(x)
            
            # Since sorted=True, topk indices should be [0, 2]
            scores = m.gate(x.view(1, -1))
            topk_scores, topk_indices = torch.topk(scores, config.num_experts_per_tok, dim=-1, sorted=True)
            assert topk_indices[0, 0].item() == 0
            assert topk_indices[0, 1].item() == 2


# ────────────────────────────────────────────────────────────────
# 4. Gradient Flow Tests
# ────────────────────────────────────────────────────────────────

class TestGPTOssFeedForwardGradients:
    def test_gradients_flow(self, config):
        m = GPTOssFeedForward(config)
        m.train()
        
        x = torch.randn(2, 4, 128, requires_grad=True)
        out = m(x)
        out.sum().backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check that experts receive gradients
        assert m.gate.weight.grad is not None
        assert not torch.isnan(m.gate.weight.grad).any()
        
        for i in range(config.num_experts):
            for name, param in m.experts[i].named_parameters():
                assert param.grad is not None, f"No grad on expert {i} param {name}"
                assert not torch.isnan(param.grad).any(), f"NaN grad on expert {i} param {name}"


# ────────────────────────────────────────────────────────────────
# 5. Block Integration tests
# ────────────────────────────────────────────────────────────────

class TestGPTOssBlockIntegration:
    def test_block_runs_with_moe(self, config):
        block = GPTOssBlock(config, layer_idx=0)
        block.eval()
        x = torch.randn(2, 8, 128)
        with torch.no_grad():
            out = block(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

# tests/test_gptoss_model.py
from dataclasses import dataclass
import pytest
import torch

from BLLMC.components.base import ModelFactory
from BLLMC.components.models import GPTOssModel


@dataclass
class GPTOssModelTestConfig:
    architecture: str = "gptoss"
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
    vocab_size: int = 1000
    n_layers: int = 2

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = torch.float32


@pytest.fixture
def config():
    return GPTOssModelTestConfig()


@pytest.fixture
def model(config):
    torch.manual_seed(42)
    m = ModelFactory.create_model(config)
    m.eval()
    return m


# ────────────────────────────────────────────────────────────────
# 1. Model Registry & Creation
# ────────────────────────────────────────────────────────────────


class TestGPTOssModelFactory:
    def test_registered(self):
        assert "gptoss" in ModelFactory._registry

    def test_factory_creation(self, model):
        assert isinstance(model, GPTOssModel)


# ────────────────────────────────────────────────────────────────
# 2. Output Shapes & Weight Tying
# ────────────────────────────────────────────────────────────────


class TestGPTOssModelForward:
    def test_output_shape(self, model):
        x = torch.randint(0, 1000, (2, 8))
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 8, 1000)

    def test_no_nan_or_inf(self, model):
        x = torch.randint(0, 1000, (2, 8))
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_weight_tying(self, model):
        # LM Head weights must be tied to Embedding weights
        assert torch.equal(model.lm_head.weight, model.embeddings.weight)

        # Modifying one must modify the other
        with torch.no_grad():
            model.embeddings.weight[0, 0] += 1.0
        assert model.lm_head.weight[0, 0] == model.embeddings.weight[0, 0]


# ────────────────────────────────────────────────────────────────
# 3. KV Cache Mechanisms
# ────────────────────────────────────────────────────────────────


class TestGPTOssModelKVCache:
    def test_model_reset_cache(self, model):
        x = torch.randint(0, 1000, (1, 4))
        with torch.no_grad():
            model(x, use_cache=True)

        # Verify cache is populated in all blocks
        for block in model.blocks:
            assert block.attn.cache_k is not None
            assert block.attn.cache_v is not None

        model.reset_cache()

        # Verify cache is cleared
        for block in model.blocks:
            assert block.attn.cache_k is None
            assert block.attn.cache_v is None
            assert block.attn.ptr_current_pos == 0

    def test_model_cache_matches_full_pass(self, model):
        x = torch.randint(0, 1000, (1, 8))

        # Full pass
        with torch.no_grad():
            out_full = model(x, use_cache=False)

        # Incremental prefill + decoding pass
        model.reset_cache()
        with torch.no_grad():
            out_prefill = model(x[:, :4], use_cache=True)
            outs_decode = []
            for i in range(4, 8):
                out_i = model(x[:, i : i + 1], use_cache=True)
                outs_decode.append(out_i)

        out_incremental = torch.cat([out_prefill] + outs_decode, dim=1)
        diff = (out_full - out_incremental).abs().max().item()
        assert diff < 1e-4, f"Incremental KV cache mismatch: {diff}"

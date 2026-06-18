"""
Unit tests for BLLMC configuration dataclasses.

Tests cover:
    - Default config creation and field values
    - Architecture preset factories (gpt2, mistral, llama3)
    - Config override mechanics
    - Validation assertions (emb_dim % n_heads, etc.)
"""

import pytest
from BLLMC.components.config import (
    GPT_Config,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    gpt2_config,
    mistral_config,
    llama3_config,
    gptoss_config,
)

# ─── Default Config ───────────────────────────────────────────────


class TestDefaultConfig:
    """Test that GPT_Config has correct default values."""

    def test_default_architecture(self):
        config = GPT_Config()
        assert config.architecture == "gpt2"

    def test_default_emb_dim(self):
        config = GPT_Config()
        assert config.emb_dim == 768

    def test_default_n_heads(self):
        config = GPT_Config()
        assert config.n_heads == 12

    def test_default_n_layers(self):
        config = GPT_Config()
        assert config.n_layers == 12

    def test_default_vocab_size(self):
        config = GPT_Config()
        assert config.vocab_size == 50257

    def test_default_context_length(self):
        config = GPT_Config()
        assert config.context_length == 256

    def test_default_learning_rate(self):
        config = GPT_Config()
        assert config.learning_rate == 5e-4

    def test_default_optimizer(self):
        config = GPT_Config()
        assert config.optimizer == "AdamW"

    def test_emb_dim_divisible_by_heads(self):
        config = GPT_Config()
        assert config.emb_dim % config.n_heads == 0

    def test_n_heads_divisible_by_kv_heads(self):
        config = GPT_Config()
        assert config.n_heads % config.n_kv_heads == 0

    def test_dtype_defaults_to_float32(self):
        import torch

        config = GPT_Config()
        assert config.dtype == torch.float32


# ─── Config Inheritance ───────────────────────────────────────────


class TestConfigInheritance:
    """Test that GPT_Config properly inherits from all base configs."""

    def test_inherits_model_config(self):
        config = GPT_Config()
        assert isinstance(config, ModelConfig)

    def test_inherits_data_config(self):
        config = GPT_Config()
        assert isinstance(config, DataConfig)

    def test_inherits_training_config(self):
        config = GPT_Config()
        assert isinstance(config, TrainingConfig)

    def test_has_model_fields(self):
        config = GPT_Config()
        assert hasattr(config, "emb_dim")
        assert hasattr(config, "n_heads")
        assert hasattr(config, "rope_base")

    def test_has_data_fields(self):
        config = GPT_Config()
        assert hasattr(config, "train_data_path")
        assert hasattr(config, "val_split")

    def test_has_training_fields(self):
        config = GPT_Config()
        assert hasattr(config, "batch_size")
        assert hasattr(config, "learning_rate")
        assert hasattr(config, "gradient_clip")


# ─── Config Overrides ─────────────────────────────────────────────


class TestConfigOverrides:
    """Test that field overrides work correctly."""

    def test_override_emb_dim(self):
        config = GPT_Config(emb_dim=512, n_heads=8)
        assert config.emb_dim == 512

    def test_override_batch_size(self):
        config = GPT_Config(batch_size=32)
        assert config.batch_size == 32

    def test_override_preserves_defaults(self):
        config = GPT_Config(emb_dim=512, n_heads=8)
        assert config.n_layers == 12  # default preserved
        assert config.vocab_size == 50257  # default preserved

    def test_invalid_emb_dim_heads_ratio(self):
        """emb_dim must be divisible by n_heads."""
        with pytest.raises(AssertionError):
            GPT_Config(emb_dim=100, n_heads=7)

    def test_invalid_heads_kv_heads_ratio(self):
        """n_heads must be divisible by n_kv_heads."""
        with pytest.raises(AssertionError):
            GPT_Config(n_heads=12, n_kv_heads=5)


# ─── Architecture Presets ─────────────────────────────────────────


class TestGPT2Preset:
    """Test gpt2_config() factory function."""

    def test_architecture_name(self):
        config = gpt2_config()
        assert config.architecture == "gpt2"

    def test_kv_heads_equal_heads(self):
        """GPT-2 uses standard MHA, so n_kv_heads == n_heads."""
        config = gpt2_config()
        assert config.n_kv_heads == config.n_heads

    def test_no_sliding_window(self):
        config = gpt2_config()
        assert config.sliding_window_size is None

    def test_override_works(self):
        config = gpt2_config(emb_dim=512, n_heads=8, n_kv_heads=8)
        assert config.emb_dim == 512
        assert config.architecture == "gpt2"


class TestMistralPreset:
    """Test mistral_config() factory function."""

    def test_architecture_name(self):
        config = mistral_config()
        assert config.architecture == "mistral"

    def test_has_sliding_window(self):
        config = mistral_config()
        assert config.sliding_window_size is not None
        assert config.sliding_window_size == 256

    def test_gqa_heads(self):
        """Mistral uses GQA with fewer KV heads."""
        config = mistral_config()
        assert config.n_kv_heads < config.n_heads

    def test_moe_params(self):
        config = mistral_config()
        assert config.num_experts == 8
        assert config.num_experts_per_tok == 2

    def test_override_experts(self):
        config = mistral_config(num_experts=4, num_experts_per_tok=1)
        assert config.num_experts == 4
        assert config.num_experts_per_tok == 1


class TestLlama3Preset:
    """Test llama3_config() factory function."""

    def test_architecture_name(self):
        config = llama3_config()
        assert config.architecture == "llama3"

    def test_high_rope_base(self):
        """LLaMA 3 uses θ=500K for extended context."""
        config = llama3_config()
        assert config.rope_base == 500_000.0

    def test_no_dropout(self):
        """LLaMA 3 doesn't use dropout."""
        config = llama3_config()
        assert config.drop_rate == 0.0

    def test_gqa_ratio(self):
        """32 query heads, 8 KV heads → 4:1 ratio."""
        config = llama3_config()
        assert config.n_heads == 32
        assert config.n_kv_heads == 8
        assert config.n_heads // config.n_kv_heads == 4

    def test_large_vocab(self):
        config = llama3_config()
        assert config.vocab_size == 128256

    def test_no_sliding_window(self):
        config = llama3_config()
        assert config.sliding_window_size is None

    def test_override_context_length(self):
        config = llama3_config(context_length=2048)
        assert config.context_length == 2048
        assert config.architecture == "llama3"  # preserved


class TestGPTOssPreset:
    """Test gptoss_config() factory function."""

    def test_architecture_name(self):
        config = gptoss_config()
        assert config.architecture == "gptoss"

    def test_vocab_size(self):
        config = gptoss_config()
        assert config.vocab_size == 201088

    def test_emb_dim(self):
        config = gptoss_config()
        assert config.emb_dim == 2880

    def test_moe_params(self):
        config = gptoss_config()
        assert config.num_experts == 32
        assert config.num_experts_per_tok == 4
        assert config.moe_hidden_dim == 2880

    def test_rope_and_swiglu_defaults(self):
        config = gptoss_config()
        assert config.rope_base == 150000.0
        assert config.sliding_window_size == 128
        assert config.swiglu_limit == 7.0
        assert config.rope_scaling_factor == 32.0
        assert config.rope_ntk_alpha == 1.0
        assert config.rope_ntk_beta == 32.0

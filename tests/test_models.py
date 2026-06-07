"""
Unit tests for the BLLMC model architectures and forward passes.

Tests cover:
    - Model instantiation via ModelFactory
    - Output tensor shape verification for all registered architectures:
        * GPT-2 (gpt2)
        * Mistral (mistral)
        * LLaMA 2 (llama2)
        * LLaMA 3 (llama3)
    - Handling of batch size, sequence length, and vocab dimensions
"""

import pytest
import torch
from BLLMC.components.config import (
    GPT_Config,
    gpt2_config,
    mistral_config,
    llama3_config,
)
from BLLMC.components.base import ModelFactory
from BLLMC.components.models import GPT2Model, MistralModel, LlamaModel, Llama3Model

# ─── Model Factory Registration ───────────────────────────────────


def test_model_factory_registration():
    """Verify all expected model names are in the registry."""
    registry = ModelFactory._registry
    assert "gpt2" in registry
    assert "mistral" in registry
    assert "llama2" in registry
    assert "llama3" in registry


# ─── GPT-2 Forward Pass ───────────────────────────────────────────


def test_gpt2_instantiation_and_forward():
    # Keep configuration small for fast testing
    config = gpt2_config(
        emb_dim=64,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        context_length=32,
        vocab_size=1000,
        batch_size=2,
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model, GPT2Model)

    # Forward pass inputs: (batch_size, sequence_length)
    x = torch.randint(0, config.vocab_size, (2, 16))
    output = model(x)

    # Expected output shape: (batch_size, sequence_length, vocab_size)
    assert output.shape == (2, 16, 1000)
    assert output.dtype == torch.float32


# ─── Mistral Forward Pass ─────────────────────────────────────────


def test_mistral_instantiation_and_forward():
    config = mistral_config(
        emb_dim=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        context_length=32,
        vocab_size=1000,
        num_experts=4,
        num_experts_per_tok=1,
        moe_hidden_dim=128,
        sliding_window_size=16,
        batch_size=2,
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model, MistralModel)

    x = torch.randint(0, config.vocab_size, (2, 16))
    output = model(x)

    assert output.shape == (2, 16, 1000)
    assert output.dtype == torch.float32


# ─── LLaMA 2 Forward Pass ─────────────────────────────────────────


def test_llama2_instantiation_and_forward():
    config = GPT_Config(
        architecture="llama2",
        emb_dim=64,
        n_heads=4,
        n_kv_heads=4,
        n_layers=2,
        context_length=32,
        vocab_size=1000,
        ffn_hidden_dim=128,
        batch_size=2,
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model, LlamaModel)

    x = torch.randint(0, config.vocab_size, (2, 16))
    output = model(x)

    assert output.shape == (2, 16, 1000)
    assert output.dtype == torch.float32


# ─── LLaMA 3 Forward Pass ─────────────────────────────────────────


def test_llama3_instantiation_and_forward():
    config = llama3_config(
        emb_dim=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        context_length=32,
        vocab_size=1000,
        ffn_hidden_dim=128,
        batch_size=2,
    )

    model = ModelFactory.create_model(config)
    assert isinstance(model, Llama3Model)

    x = torch.randint(0, config.vocab_size, (2, 16))
    output = model(x)

    assert output.shape == (2, 16, 1000)
    assert output.dtype == torch.float32


# ─── Edge Cases & Verification ────────────────────────────────────


def test_unsupported_architecture_raises():
    config = GPT_Config(architecture="unknown_model")
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        ModelFactory.create_model(config)


def test_forward_pass_batch_size_one():
    config = gpt2_config(
        emb_dim=32,
        n_heads=2,
        n_kv_heads=2,
        n_layers=1,
        context_length=16,
        vocab_size=500,
        batch_size=1,
    )
    model = ModelFactory.create_model(config)
    x = torch.randint(0, config.vocab_size, (1, 8))
    output = model(x)
    assert output.shape == (1, 8, 500)

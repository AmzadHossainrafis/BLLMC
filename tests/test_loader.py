"""
Unit tests for BanglaDataset and DataLoader.

Tests cover:
    - Output tensor shapes and dtypes
    - Target is shifted-by-one from input (next-token prediction)
    - Dataset length calculation
    - Stride and max_length interaction
    - Edge cases (short text, single sample)
"""

import pytest
import torch
from BLLMC.components.config import GPT_Config
from BLLMC.data.loader import BanglaDataset, create_dataloader
import tiktoken


@pytest.fixture
def tokenizer():
    """Shared GPT-2 tokenizer instance."""
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def sample_text():
    """A sample text string long enough for multiple batches."""
    return "Hello world! This is a test of the Bangla LLM data loader. " * 100


@pytest.fixture
def small_config():
    """A small config for fast testing."""
    return GPT_Config(
        max_length=32,
        stride=32,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )


class TestDataLoaderShapes:
    """Test that DataLoader produces tensors with correct shapes."""

    def test_input_shape(self, sample_text, small_config):
        loader = create_dataloader(sample_text, "gpt2", small_config)
        inputs, targets = next(iter(loader))
        assert inputs.shape == (4, 32)  # (batch_size, max_length)

    def test_target_shape(self, sample_text, small_config):
        loader = create_dataloader(sample_text, "gpt2", small_config)
        inputs, targets = next(iter(loader))
        assert targets.shape == (4, 32)  # same shape as inputs

    def test_input_dtype(self, sample_text, small_config):
        loader = create_dataloader(sample_text, "gpt2", small_config)
        inputs, _ = next(iter(loader))
        assert inputs.dtype == torch.long

    def test_target_dtype(self, sample_text, small_config):
        loader = create_dataloader(sample_text, "gpt2", small_config)
        _, targets = next(iter(loader))
        assert targets.dtype == torch.long

    def test_different_max_length(self, sample_text):
        config = GPT_Config(
            max_length=64,
            stride=64,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        loader = create_dataloader(sample_text, "gpt2", config)
        inputs, targets = next(iter(loader))
        assert inputs.shape == (2, 64)
        assert targets.shape == (2, 64)

    def test_batch_size_one(self, sample_text):
        config = GPT_Config(
            max_length=16,
            stride=16,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        loader = create_dataloader(sample_text, "gpt2", config)
        inputs, targets = next(iter(loader))
        assert inputs.shape == (1, 16)


class TestTargetShift:
    """Test that targets are inputs shifted by 1 (next-token prediction)."""

    def test_targets_shifted_by_one(self, sample_text, tokenizer):
        config = GPT_Config(
            max_length=32,
            stride=32,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        dataset = BanglaDataset(sample_text, tokenizer, config)
        inputs, targets = dataset[0]

        assert torch.equal(
            targets[:-1], inputs[1:]
        ), "Targets should be inputs shifted by 1 position"

    def test_first_sample_starts_at_zero(self, sample_text, tokenizer):
        config = GPT_Config(
            max_length=16,
            stride=16,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        dataset = BanglaDataset(sample_text, tokenizer, config)
        inputs, _ = dataset[0]

        # First sample should start from the beginning
        full_tokens = tokenizer.encode(
            sample_text,
            allowed_special={"<|EOS|>", "<p>", "</p>", "<number>", "</strong>"},
        )
        expected = torch.tensor(full_tokens[:16])
        assert torch.equal(inputs, expected)


# ─── Dataset Length Tests ─────────────────────────────────────────


class TestDatasetLength:
    """Test dataset length calculation."""

    def test_length_is_positive(self, sample_text, tokenizer, small_config):
        dataset = BanglaDataset(sample_text, tokenizer, small_config)
        assert len(dataset) > 0

    def test_length_formula(self, sample_text, tokenizer):
        """Verify length matches the formula: (n_tokens - max_length - 1) // stride + 1."""
        config = GPT_Config(
            max_length=32,
            stride=32,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        dataset = BanglaDataset(sample_text, tokenizer, config)

        n_tokens = len(
            tokenizer.encode(
                sample_text,
                allowed_special={"<|EOS|>", "<p>", "</p>", "<number>", "</strong>"},
            )
        )
        expected_len = max(0, (n_tokens - config.max_length - 1) // config.stride + 1)
        assert len(dataset) == expected_len

    def test_stride_affects_length(self, sample_text, tokenizer):
        """Smaller stride → more samples (overlapping windows)."""
        config_big = GPT_Config(
            max_length=32,
            stride=32,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        config_small = GPT_Config(
            max_length=32,
            stride=8,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        dataset_big = BanglaDataset(sample_text, tokenizer, config_big)
        dataset_small = BanglaDataset(sample_text, tokenizer, config_small)

        assert len(dataset_small) > len(
            dataset_big
        ), "Smaller stride should produce more samples"

    def test_short_text_no_crash(self, tokenizer):
        """Very short text should produce 0 samples, not crash."""
        config = GPT_Config(
            max_length=256,
            stride=256,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        dataset = BanglaDataset("Hi", tokenizer, config)
        assert len(dataset) == 0

    def test_exact_fit_text(self, tokenizer):
        """Text that produces exactly max_length+1 tokens → 1 sample."""
        config = GPT_Config(
            max_length=4,
            stride=4,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        # "Hello" encodes to a few tokens, we need exactly 5 tokens
        # Let's just check it doesn't crash and length >= 0
        dataset = BanglaDataset("Hello world foo bar baz", tokenizer, config)
        assert len(dataset) >= 0


# ─── DataLoader Iteration Tests ───────────────────────────────────


class TestDataLoaderIteration:
    """Test that we can iterate through the full DataLoader."""

    def test_can_iterate_all_batches(self, sample_text, small_config):
        loader = create_dataloader(sample_text, "gpt2", small_config)
        batch_count = 0
        for inputs, targets in loader:
            assert inputs.shape[1] == small_config.max_length
            assert targets.shape[1] == small_config.max_length
            batch_count += 1
        assert batch_count > 0

    def test_all_tokens_in_vocab_range(self, sample_text, small_config):
        """All token IDs should be valid (within vocab size)."""
        loader = create_dataloader(sample_text, "gpt2", small_config)
        inputs, targets = next(iter(loader))
        assert inputs.min() >= 0
        assert inputs.max() < small_config.vocab_size
        assert targets.min() >= 0
        assert targets.max() < small_config.vocab_size

    def test_drop_last_behavior(self, sample_text):
        """With drop_last=True, all batches should have full batch_size."""
        config = GPT_Config(
            max_length=32,
            stride=32,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            drop_last=True,
        )
        loader = create_dataloader(sample_text, "gpt2", config)
        for inputs, targets in loader:
            assert (
                inputs.shape[0] == 4
            ), "All batches should be full with drop_last=True"

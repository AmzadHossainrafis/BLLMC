# src/BLLMC/components/config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Base model architecture config shared by all architectures."""

    architecture: str = "gpt2"
    vocab_size: int = 50257
    emb_dim: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    n_layers: int = 12
    context_length: int = 256
    drop_rate: float = 0.1
    rope_base: float = 100_000.0
    dtype: object = None  # defaults to torch.float32 in __post_init__

    # ── Sliding Window Attention ──
    sliding_window_size: Optional[int] = None  # None = full causal attention

    # ── MoE (Mixture of Experts) ──
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_hidden_dim: int = 768

    def __post_init__(self):
        import torch

        if self.dtype is None:
            self.dtype = torch.float32

        assert (
            self.emb_dim % self.n_heads == 0
        ), f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"


@dataclass
class DataConfig:
    """Data paths and split ratios."""

    dataset_path: str = "/home/rafi/Desktop/BLLMC/dataset/demo_text.txt"
    train_data_path: str = "dataset/english_data.txt"
    val_data_path: str = "dataset/english_val.txt"
    test_data_path: str = "dataset/bangla_test.txt"
    val_split: float = 0.1
    test_split: float = 0.1
    train_split: float = 0.9


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 17
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    max_epochs: int = 10
    warmup_steps: int = 100
    checkpoint_dir: str = "artifacts/model_ckpt"
    eval_iters: int = 10
    eval_interval: int = 50
    start_context: str = ""
    optimizer: str = "AdamW"
    gradient_clip: float = 1.0
    compile: bool = False
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = True
    max_length: int = 256
    stride: int = 256
    gradient_checkpointing: bool = False
    gen_indx: int = 5


@dataclass
class GPT_Config(ModelConfig, DataConfig, TrainingConfig):
    """
    Full configuration combining model, data, and training settings.

    This is the single config object passed throughout the codebase.
    It inherits from the three composable base configs so fields can
    be referenced individually when needed, while maintaining full
    backward compatibility with existing code that uses GPT_Config.
    """

    pass


# ── Architecture Presets ──


def gpt2_config(**overrides) -> GPT_Config:
    """GPT-2 style: LayerNorm, learned positional embeddings, dense FFN."""
    defaults = dict(
        architecture="gpt2",
        n_heads=12,
        n_kv_heads=12,
        sliding_window_size=None,
    )
    defaults.update(overrides)
    return GPT_Config(**defaults)


def mistral_config(**overrides) -> GPT_Config:
    """Mistral style: RMSNorm, RoPE, sliding window attention, MoE FFN."""
    defaults = dict(
        architecture="mistral",
        n_heads=12,
        n_kv_heads=4,
        sliding_window_size=4096,
        num_experts=8,
        num_experts_per_tok=2,
        moe_hidden_dim=768,
    )
    defaults.update(overrides)
    return GPT_Config(**defaults)

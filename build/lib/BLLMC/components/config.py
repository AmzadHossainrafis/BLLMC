# src/BLLMC/components/config.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoPEConfig:
    """RoPE frequency scaling configuration."""
    rope_base: float = 500_000.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_context_length: int = 8192


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    vocab_size: int = 50257
    emb_dim: int = 768
    n_heads: int = 12
    n_kv_heads: Optional[int] = None  # None = standard MHA, int = GQA
    n_layers: int = 12
    context_length: int = 256
    drop_rate: float = 0.1
    rope: RoPEConfig = field(default_factory=RoPEConfig)

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads  # default: MHA
        assert self.emb_dim % self.n_heads == 0, \
            f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    num_epochs: int = 10
    warmup_steps: int = 100
    checkpoint_dir: str = "artifacts/model_ckpt"


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_path: str = "dataset/bangla_news.txt"
    train_split: float = 0.9
    max_length: int = 256

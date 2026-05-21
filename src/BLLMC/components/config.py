# src/BLLMC/components/config.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPT_Config:
    architecture: str = "gpt2"
    vocab_size: int = 50257
    emb_dim: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    n_layers: int = 12
    context_length: int = 256
    drop_rate: float = 0.1
    rope_base: float = 100_000.0

    # ── Data ──
    dataset_path: str = "/home/rafi/Desktop/BLLMC/dataset/demo_text.txt"
    train_data_path: str = "dataset/english_data.txt"
    val_data_path: str = "dataset/english_val.txt"
    test_data_path: str = "dataset/bangla_test.txt"
    val_split: float = 0.1
    test_split: float = 0.1
    train_split: float = 0.9

    # ── Training ──
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

    def __post_init__(self):
        assert (
            self.emb_dim % self.n_heads == 0
        ), f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

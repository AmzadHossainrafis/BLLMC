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
    dtype: object = None
    sliding_window_size: Optional[int] = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_hidden_dim: int = 768
    ffn_hidden_dim: int = 3072
    swiglu_limit: float = 7.0
    rope_scaling_factor: float = 1.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0

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

    dataset_path: str = "dataset/demo_text.txt"
    train_data_path: str = "dataset/english_data.txt"
    val_data_path: str = "dataset/english_val.txt"
    test_data_path: str = "dataset/bangla_test.txt"
    val_split: float = 0.1
    test_split: float = 0.1
    train_split: float = 0.9
    tokenizer_backend: str = "sentencepiece"
    tokenizer_model: str = "./dataset/tokenizer_model/tokenizer.model"
    hf_token: Optional[str] = None
    repo_id: str = "hf-internal-testing/llama-tokenizer"
    filename: str = "tokenizer.model"
    local_path: str = "./dataset/tokenizer_model"


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
    gen_strategy: str = "greedy"
    gen_temperature: float = 1.0
    gen_top_k: int = 50


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
        sliding_window_size=256,
        num_experts=8,
        num_experts_per_tok=2,
        moe_hidden_dim=768,
    )
    defaults.update(overrides)
    return GPT_Config(**defaults)


def llama3_config(**overrides) -> GPT_Config:
    """LLaMA 3 style: RMSNorm, RoPE, GQA, SwiGLU FFN."""
    defaults = dict(
        architecture="llama3",
        n_heads=32,
        n_kv_heads=8,
        emb_dim=4096,
        n_layers=32,
        context_length=8192,
        rope_base=500_000.0,
        ffn_hidden_dim=14336,
        vocab_size=128256,
        sliding_window_size=None,
        drop_rate=0.0,  # LLaMA 3 doesn't use dropout
        tokenizer_backend="tiktoken",
        tokenizer_model="cl100k_base",
        hf_token=None,
        repo_id="openai-gpt",
        filename="vocab.json",
        local_path="./dataset/tokenizer_model",
    )
    defaults.update(overrides)
    return GPT_Config(**defaults)


def llama2_config(**overrides) -> GPT_Config:
    """LLaMA 2 style: RMSNorm, RoPE, GQA, SwiGLU FFN."""
    defaults = dict(
        architecture="llama2",
        n_heads=32,
        n_kv_heads=8,
        emb_dim=4096,
        n_layers=32,
        context_length=2048,
        rope_base=500_000.0,
        ffn_hidden_dim=14336,
        vocab_size=32000,
        sliding_window_size=None,
        drop_rate=0.0,  # LLaMA 2 doesn't use dropout
        tokenizer_backend="sentencepiece",
        tokenizer_model="./dataset/tokenizer_model/tokenizer.model",
        hf_token=None,
        repo_id="hf-internal-testing/llama-tokenizer",
        filename="tokenizer.model",
        local_path="./dataset/tokenizer_model",
    )
    defaults.update(overrides)
    return GPT_Config(**defaults)


def gptoss_config(**overrides) -> GPT_Config:
    """GPT-OSS style: GQA, sliding window, learnable attention sinks, custom SwiGLU MoE."""
    defaults = dict(
        architecture="gptoss",
        vocab_size=201088,
        emb_dim=2880,
        n_heads=64,
        n_kv_heads=8,
        n_layers=24,
        context_length=4096,
        rope_base=150000.0,
        sliding_window_size=128,
        num_experts=32,
        num_experts_per_tok=4,
        moe_hidden_dim=2880,
        swiglu_limit=7.0,
        rope_scaling_factor=32.0,
        rope_ntk_alpha=1.0,
        rope_ntk_beta=32.0,
        drop_rate=0.0,
        tokenizer_backend="tiktoken",
        tokenizer_model="o200k_harmony",
        hf_token=None,
        repo_id="openai-gpt",
        filename="vocab.json",
        local_path="./dataset/tokenizer_model",
    )
    defaults.update(overrides)
    return GPT_Config(**defaults)

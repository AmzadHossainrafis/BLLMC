# 🇧🇩 BLLMC: Bangla LLM Collection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A highly modular, extensible, and high-performance PyTorch library for training and deploying **Bangla Language Models (LLMs)** from scratch. Built with a focus on modern LLM innovations including **Grouped Query Attention (GQA)**, **Sliding Window Attention (SWA)**, **Rotary Position Embeddings (RoPE)**, **Sparse Mixture of Experts (MoE)**, and **Learnable Attention Sinks**.

> [!TIP]
> 🌟 **Don't forget to star the repository if you find it helpful!** Looking to collaborate? Feel free to reach out or open a Pull Request.

---

## ✨ Features

- 🏗️ **Modular & Composable Architecture**: Decoupled attention mechanisms, positional embeddings, normalizations, and feedforward blocks that can be easily plugged into any model.
- ⚡ **State-of-the-Art Core Layers**:
  - **Rotary Position Embeddings (RoPE)**: High-precision relative positional encoding with configurable base frequency, NTK-aware frequency scaling, and dynamic context length expansion.
  - **Grouped Query Attention (GQA)**: Memory-efficient attention with shared KV heads (matching LLaMA 3 / GPT-OSS).
  - **Sliding Window Attention (SWA)**: Bounded memory footprint for long-context modeling with KV caching (matching Mistral).
  - **Learnable Attention Sinks**: Preserves long-context performance by allowing queries to attend to persistent learnable sink tokens.
  - **Sparse Mixture of Experts (MoE)**: Gated sparse routing with top-k expert selection and custom SwiGLU activation.
  - **Fused Operations**: Integrated with PyTorch's `scaled_dot_product_attention` for FlashAttention compatibility.
- 🏭 **Model Factory & Registry**: Easily register and instantiate architectures (`gpt2`, `mistral`, `llama2`, `llama3`, `gptoss`) via a dynamic `@ModelFactory.register` decorator pattern.
- 🔠 **Tokenizer Strategy Pattern**: Pluggable `sentencepiece` and `tiktoken` backends supporting automatic downloading of models from the Hugging Face Hub.
- 🎮 **Pluggable Generation Strategies**: Strategy Pattern implementations for token sampling, including `Greedy`, `Temperature-based`, and `Top-K` strategies.
- 📊 **Bangla-Optimized Data Pipeline**: Built-in corpus downloader, automated train/val/test splitter, and windowed token-by-token sequence generation loaders.
- 🚀 **Trainer Design Pattern**: A scalable LLM trainer supporting AMP (automatic mixed precision), Cosine Warmup with cosine annealing scheduler, gradient clipping, gradient accumulation, validation monitoring, checkpointing, and interactive text-generation during training.

---

## 📁 Repository Structure

```
BLLMC/
├── src/BLLMC/                 # Core Source Package
│   ├── components/            # Neural Network Blocks & Modules
│   │   ├── attention/         # Attention Implementations
│   │   │   ├── multi_head.py          # Multi-Head Attention + Multi-Head Attention with RoPE
│   │   │   ├── grouped_query.py       # Grouped-Query Attention (GQA) with RoPE & KV Cache
│   │   │   ├── sw_attention.py        # Sliding Window Attention with KV Caching & RoPE
│   │   │   ├── gqa_sliding_window.py  # Unified GQA + Sliding Window Attention with SDPA support
│   │   │   ├── gptoss_attention.py    # GPT-OSS Attention (GQA + sinks + sliding window)
│   │   │   └── multihead_latent.py    # Multihead Latent Attention (MLA)
│   │   │
│   │   ├── layers/            # Neural Network Layers
│   │   │   ├── embeddings.py          # RoPE computation and offset applications
│   │   │   ├── feedforward.py         # Standard FeedForward, MoE, SwiGLU, and GPTOssFeedForward
│   │   │   ├── normalization.py       # LayerNorm & RMSNorm implementations
│   │   │   └── activations.py         # Custom Activations (GELU, SwiGLU with clamping & bias)
│   │   │
│   │   ├── blocks/            # Unified Transformer Blocks
│   │   │   ├── gpt2_block.py          # LayerNorm + MultiHeadAttention + FeedForward
│   │   │   ├── mistral_block.py       # RMSNorm + SWA + MoE FeedForward
│   │   │   ├── llama_block.py         # Llama2Block (MHA+RoPE) & Llama3Block (GQA+SwiGLU)
│   │   │   └── gptoss_block.py        # GPTOssBlock (RMSNorm + Alternating SWA + MoE SwiGLU)
│   │   │
│   │   ├── base.py            # CosineWarmupScheduler, Abstract Trainer & ModelFactory base
│   │   ├── config.py          # Composable Dataclass Configs with Architecture Presets
│   │   ├── models.py          # Model Registry & Implementations (GPT2, Mistral, Llama2, Llama3, GPTOss)
│   │   ├── generation.py      # Generation Strategy Pattern (Greedy, Temperature, Top-K)
│   │   └── trainer.py         # Concrete LLMTrainer with Gradient Accumulation & AMP
│   │
│   ├── data/                  # Dataset and Dataloading Pipeline
│   │   ├── dataset.py         # Google Drive Corpus Downloader (Bangla dataset v3)
│   │   ├── ingestion.py       # Raw Data Ingestion & Train/Val/Test Splitter
│   │   ├── loader.py          # Token-level Sliding Window DataLoader
│   │   └── tokenizer.py       # TokenizerStrategy (SentencePiece and Tiktoken strategies)
│   │
│   ├── pipeline/              # High-Level Execution Pipelines
│   │   ├── train_pipeline.py  # End-to-end Training Script
│   │   └── inference_pipeline.py # Autoregressive Generation
│   │
│   └── utils/                 # General Purpose Utilities
│       ├── common.py          # Configuration YAML parsers
│       ├── logger.py          # Logging systems
│       └── exception.py       # Custom Exception structures
│
├── config/                    # YAML Configuration files
├── artifacts/                 # Checkpoints, Model Weights
├── notebook/                  # Research, EDA, & Trial notebooks
├── tests/                     # Unit & Integration Tests (covering configs, models, attention, generation)
├── inference.py               # CLI tool to run inference on checkpoints
├── setup.py                   # Packaging & Distribution configuration
└── pyproject.toml             # Project dependency configuration
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the package along with its dependencies:

```bash
git clone https://github.com/AmzadHossainrafis/BLLMC
cd BLLMC
pip install -r requirements.txt
pip install -e .
```

### 2. Download the Bangla Dataset

We provide an automated utility to download a rich Bangla raw text corpus (v3) directly from Google Drive:

```python
from BLLMC.data.dataset import download_dataset

# Downloads and saves the dataset under dataset/bangla_dataset.txt
download_dataset()
```

### 3. Running Data Ingestion

Split your raw text corpus into training, validation, and test datasets automatically using your configuration:

```python
from BLLMC.components.config import GPT_Config
from BLLMC.data.ingestion import DataIngestion

config = GPT_Config()
ingestion = DataIngestion(config)
train_path, val_path, test_path = ingestion.initiate_data_ingestion()
print(f"Data split saved: Train -> {train_path}, Val -> {val_path}")
```

### 4. Training a Model

Train a model from scratch using the high-level train pipeline. Switch architectures easily by changing the config preset:

```python
from BLLMC.components.config import gptoss_config
from BLLMC.components.models import ModelFactory
from BLLMC.data.loader import create_dataloader
from BLLMC.components.trainer import LLMTrainer

# 1. Choose your architecture (gpt2, mistral, llama2, llama3, gptoss)
config = gptoss_config(
    batch_size=4,
    gradient_accumulation_steps=8,
    max_epochs=5,
    n_layers=12,
    emb_dim=512,
    n_heads=8,
    n_kv_heads=2,
)

# 2. Build the model via the registry
model = ModelFactory.create_model(config)

# 3. Read data splits
with open(config.train_data_path, "r", encoding="utf-8") as f:
    train_data = f.read()
with open(config.val_data_path, "r", encoding="utf-8") as f:
    val_data = f.read()

# 4. Create DataLoaders
train_loader = create_dataloader(train_data, config.tokenizer_backend, config)
val_loader = create_dataloader(val_data, config.tokenizer_backend, config)

# 5. Train
trainer = LLMTrainer(model, train_loader, val_loader, config)
trainer.train()
```

### 5. Running Inference

You can run text generation on a trained model checkpoint using the provided `inference.py` script:

```bash
python inference.py \
    --checkpoint artifacts/model_ckpt/ckpt_epoch_4.pt \
    --prompt "বাধা দিয়ে ডাক্তারবাবু বলেন," \
    --strategy top_k \
    --temperature 0.8 \
    --top_k 50
```

---

## 🧠 Supported Architectures

| Architecture | Attention | Positional Encoding | FFN | Normalization |
|-------------|-----------|-------------------|-----|---------------|
| **GPT-2** | Multi-Head Attention | Learned Positional Embeddings | Dense (GELU) | LayerNorm |
| **Mistral** | Sliding Window Attention + RoPE | RoPE (θ=100K) | Sparse MoE (SwiGLU) | RMSNorm |
| **LLaMA 2** | Multi-Head Attention + RoPE | RoPE (θ=100K) | SwiGLU | RMSNorm |
| **LLaMA 3** | Grouped Query Attention + RoPE | RoPE (θ=500K) | SwiGLU | RMSNorm |
| **GPT-OSS** | Grouped Query Attention + Sinks + Alternating Sliding Window | RoPE (θ=150K) + NTK Scaling (α=1, β=32) | Custom Sparse MoE (SwiGLU with Clamping) | RMSNorm |

### GPT-2 Style
- Dense FeedForward layers with GELU activation
- Learned absolute positional embeddings
- Standard LayerNorm and causal Multi-Head Attention
- Weight tying between token embeddings and LM head

### Mistral Style
- **RoPE**: Rotary Position Embeddings for dynamic sequence extrapolation
- **SWA**: Sliding Window Attention with configurable window size and KV caching
- **MoE**: Sparse Mixture of Experts with gated SwiGLU feedforward routing
- Fused QKV projection with FlashAttention support via `F.scaled_dot_product_attention`

### LLaMA 2 Style
- **RoPE**: Rotary Position Embeddings applied in (B, H, T, D) layout
- **Multi-Head Attention** with full KV caching for autoregressive generation
- **SwiGLU** feedforward (gate + up projection → SiLU → down projection)
- Pre-normalization with RMSNorm

### LLaMA 3 Style
- **GQA**: Grouped Query Attention with configurable KV head groups (default 8 KV heads for 32 query heads)
- **RoPE**: High-frequency base (θ=500K) for extended context support
- **SwiGLU** feedforward matching LLaMA 2 architecture
- No dropout (following Meta's official design)

### GPT-OSS Style
- **Alternating Sliding Window**: Alternates local attention windows (e.g. size 128) and full context windows across transformer blocks.
- **Learnable Attention Sinks**: Incorporates learnable query-sink token parameters to stabilize long-context attention computation.
- **Dynamic RoPE**: Frequency base (θ=150K) coupled with NTK-aware scaling factors and interpolation bounds.
- **Custom SwiGLU MoE**: Highly parameterized Mixture of Experts with gated activations, custom +1 linear bias, and clamping limit parameters (default 7.0) to prevent numerical instability.

---

## ⚙️ Configuration

All architecture, data, and training settings are managed through composable dataclass configs:

```python
from BLLMC.components.config import gpt2_config, mistral_config, llama3_config, gptoss_config

# Architecture-specific presets with overrides
config = gpt2_config(emb_dim=512, n_layers=6)
config = mistral_config(sliding_window_size=256, num_experts=8)
config = llama3_config(n_heads=32, n_kv_heads=8, rope_base=500_000.0)
config = gptoss_config(num_experts=32, swiglu_limit=7.0, rope_scaling_factor=32.0)
```

### Key Configuration Fields

| Field | Default | Description |
|-------|---------|-------------|
| `architecture` | `"gpt2"` | Model architecture (`gpt2`, `mistral`, `llama2`, `llama3`, `gptoss`) |
| `emb_dim` | `768` | Embedding dimension |
| `n_heads` | `12` | Number of attention heads |
| `n_kv_heads` | `4` | Number of KV heads (for GQA) |
| `n_layers` | `12` | Number of transformer blocks |
| `context_length` | `256` | Maximum sequence length |
| `drop_rate` | `0.1` | Dropout rate (0.0 for LLaMA & GPT-OSS) |
| `rope_base` | `100,000` | RoPE frequency base |
| `rope_scaling_factor`| `1.0` | RoPE scaling factor for context extension |
| `rope_ntk_alpha` | `1.0` | NTK scaling alpha parameter |
| `rope_ntk_beta` | `32.0` | NTK scaling beta parameter |
| `vocab_size` | `50,257` | Vocabulary size |
| `num_experts` | `8` | Number of MoE experts |
| `num_experts_per_tok` | `2` | Top-k expert routing |
| `moe_hidden_dim` | `768` | Expert FFN hidden dimension |
| `ffn_hidden_dim` | `3,072` | Standard/SwiGLU hidden dimension |
| `swiglu_limit` | `7.0` | Max clamping limit for SwiGLU activation |
| `batch_size` | `17` | Training batch size (per micro-batch) |
| `gradient_accumulation_steps` | `16` | Number of steps to accumulate gradients before optimization step |
| `learning_rate` | `5e-4` | Learning rate |
| `warmup_steps` | `100` | Warmup steps for Cosine Warmup Scheduler |
| `min_lr` | `1e-5` | Minimum learning rate for cosine annealing |
| `gradient_clip` | `1.0` | Gradient clipping norm |
| `compile` | `False` | Compile model via `torch.compile()` |
| `tokenizer_backend` | `"sentencepiece"` | Pluggable tokenizer backend (`sentencepiece`, `tiktoken`) |
| `tokenizer_model` | `"tokenizer.model"`| Tokenizer model name/path (e.g. `cl100k_base`, `o200k_harmony`, or local path) |

---

## 🗺️ Roadmap & Future Integrations

- [x] **GPT-2** architecture
- [x] **Mistral** architecture with SWA + MoE
- [x] **LLaMA 2** architecture with RoPE
- [x] **LLaMA 3** architecture with GQA
- [x] **GPT-OSS** architecture with sinks, alternating window attention, and SwiGLU MoE
- [x] **Tokenizer Strategy Pattern** — pluggable `tiktoken` / `SentencePiece` backends
- [x] **Pluggable Generation Strategies** — Strategy Pattern for Greedy, Temperature, and Top-K sampling
- [x] **Gradient Accumulation** support in LLMTrainer
- [ ] **Distributed Multi-GPU (DDP)** and FSDP training pipelines
- [ ] **Auxiliary Gating Load-Balancing Loss** for MoE stability
- [ ] **Gemma** architecture integration
- [ ] **Qwen 3** architecture integration

---

## 🤝 Contributors & Support

This project is created and maintained by **[Amzad Hossain Rafi](mailto:amzadhossain880@gmail.com)**.

If you are interested in collaborating, scaling up Bangla LLM capabilities, or have suggestions, feel free to **ping me or open an issue/PR!**

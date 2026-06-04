# 🇧🇩 BLLMC: Bangla LLM Collection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A highly modular, extensible, and high-performance PyTorch library for training and deploying **Bangla Language Models (LLMs)** from scratch. Built with a focus on modern LLM innovations including **Grouped Query Attention (GQA)**, **Sliding Window Attention (SWA)**, **Rotary Position Embeddings (RoPE)**, and **Sparse Mixture of Experts (MoE)**.

> [!TIP]
> 🌟 **Don't forget to star the repository if you find it helpful!** Looking to collaborate? Feel free to reach out or open a Pull Request.

---

## ✨ Features

- 🏗️ **Modular & Composable Architecture**: Decoupled attention mechanisms, positional embeddings, normalizations, and feedforward blocks that can be easily plugged into any model.
- ⚡ **State-of-the-Art Core Layers**:
  - **Rotary Position Embeddings (RoPE)**: High-precision relative positional encoding with configurable base frequency.
  - **Grouped Query Attention (GQA)**: Memory-efficient attention with shared KV heads, as used in LLaMA 3.
  - **Sliding Window Attention (SWA)**: Bounded memory footprint for long-context modeling with KV caching.
  - **Sparse Mixture of Experts (MoE)**: Gated sparse routing with top-k expert selection and SwiGLU activation.
  - **Fused Operations**: Integrated with PyTorch's `scaled_dot_product_attention` for FlashAttention compatibility.
- 🏭 **Model Factory & Registry**: Easily instantiate architectures (`gpt2`, `mistral`, `llama2`, `llama3`) via a dynamic registration pattern.
- 📊 **Bangla-Optimized Data Pipeline**: Built-in corpus downloader, automated train/val/test splitter, and windowed token-by-token sequence generation loaders.
- 🚀 **Trainer Design Pattern**: A scalable LLM trainer supporting AMP (mixed precision), gradient clipping, validation monitoring, checkpointing, and interactive text-generation during training.

---

## 📁 Repository Structure

```
BLLMC/
├── src/BLLMC/                 # Core Source Package
│   ├── components/            # Neural Network Blocks & Modules
│   │   ├── attention/         # Attention Implementations
│   │   │   ├── multi_head.py      # Multi-Head Attention + Multi-Head Attention with RoPE
│   │   │   ├── grouped_query.py   # Grouped-Query Attention (GQA) with RoPE & KV Cache
│   │   │   ├── sw_attention.py    # Sliding Window Attention with KV Caching & RoPE
│   │   │   └── multihead_latent.py # Multihead Latent Attention (MLA)
│   │   │
│   │   ├── layers/            # Neural Network Layers
│   │   │   ├── embeddings.py      # RoPE (Rotary Position Embeddings) computation
│   │   │   ├── feedforward.py     # FeedForward, MoE FeedForward, SwiGLU FeedForward
│   │   │   ├── normalization.py   # LayerNorm & RMSNorm implementations
│   │   │   └── activations.py     # Custom Activations (GELU)
│   │   │
│   │   ├── blocks/            # Unified Transformer Blocks
│   │   │   ├── gpt2_block.py      # LayerNorm + MultiHeadAttention + FeedForward
│   │   │   ├── mistral_block.py   # RMSNorm + SWA + MoE FeedForward
│   │   │   └── llama_block.py     # Llama2Block (MHA+RoPE) & Llama3Block (GQA+SwiGLU)
│   │   │
│   │   ├── config.py          # Composable Dataclass Configs with Architecture Presets
│   │   ├── models.py          # Model Registry & Factory (GPT2, Mistral, Llama2, Llama3)
│   │   └── trainer.py         # Abstract Trainer & LLMTrainer with AMP support
│   │
│   ├── data/                  # Dataset and Dataloading Pipeline
│   │   ├── dataset.py         # Google Drive Corpus Downloader
│   │   ├── ingestion.py       # Raw Data Ingestion & Train/Val/Test Splitter
│   │   └── loader.py          # Token-level Sliding Window DataLoader
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
├── notebook/                  # Research & Trial notebooks
├── tests/                     # Unit & Integration Tests
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

We provide an automated utility to download a rich Bangla raw text corpus directly from Google Drive:

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

Train a model from scratch using the high-level train pipeline. Switch architectures by changing the config preset:

```python
from BLLMC.components.config import GPT_Config, mistral_config, llama3_config
from BLLMC.components.models import ModelFactory
from BLLMC.data.loader import create_dataloader
from BLLMC.components.trainer import LLMTrainer

# 1. Choose your architecture (gpt2, mistral, llama2, llama3)
config = mistral_config(
    batch_size=8,
    max_epochs=5,
    num_experts=8,             # Total experts in MoE FFN
    num_experts_per_tok=2,     # Top-2 sparse routing
)

# 2. Build the model via the registry
model = ModelFactory.create_model(config)

# 3. Read data splits
with open(config.train_data_path, "r", encoding="utf-8") as f:
    train_data = f.read()
with open(config.val_data_path, "r", encoding="utf-8") as f:
    val_data = f.read()

# 4. Create DataLoaders
train_loader = create_dataloader(train_data, "gpt2", config)
val_loader = create_dataloader(val_data, "gpt2", config)

# 5. Train
trainer = LLMTrainer(model, train_loader, val_loader, config)
trainer.train()
```

---

## 🧠 Supported Architectures

| Architecture | Attention | Positional Encoding | FFN | Normalization |
|-------------|-----------|-------------------|-----|---------------|
| **GPT-2** | Multi-Head Attention | Learned Positional Embeddings | Dense (GELU) | LayerNorm |
| **Mistral** | Sliding Window Attention + RoPE | RoPE (θ=100K) | Sparse MoE (SwiGLU) | RMSNorm |
| **LLaMA 2** | Multi-Head Attention + RoPE | RoPE (θ=100K) | SwiGLU | RMSNorm |
| **LLaMA 3** | Grouped Query Attention + RoPE | RoPE (θ=500K) | SwiGLU | RMSNorm |

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

---

## ⚙️ Configuration

All architecture, data, and training settings are managed through composable dataclass configs:

```python
from BLLMC.components.config import GPT_Config, gpt2_config, mistral_config, llama3_config

# Default config
config = GPT_Config()

# Architecture-specific presets with overrides
config = gpt2_config(emb_dim=512, n_layers=6)
config = mistral_config(sliding_window_size=256, num_experts=8)
config = llama3_config(n_heads=32, n_kv_heads=8, rope_base=500_000.0)
```

### Key Configuration Fields

| Field | Default | Description |
|-------|---------|-------------|
| `architecture` | `"gpt2"` | Model architecture (`gpt2`, `mistral`, `llama2`, `llama3`) |
| `emb_dim` | `768` | Embedding dimension |
| `n_heads` | `12` | Number of attention heads |
| `n_kv_heads` | `4` | Number of KV heads (for GQA) |
| `n_layers` | `12` | Number of transformer blocks |
| `context_length` | `256` | Maximum sequence length |
| `drop_rate` | `0.1` | Dropout rate (0.0 for LLaMA 3) |
| `rope_base` | `100,000` | RoPE frequency base |
| `vocab_size` | `50,257` | Vocabulary size |
| `num_experts` | `8` | Number of MoE experts |
| `num_experts_per_tok` | `2` | Top-k expert routing |
| `ffn_hidden_dim` | `3,072` | SwiGLU hidden dimension |
| `batch_size` | `17` | Training batch size |
| `learning_rate` | `5e-4` | Learning rate |
| `max_epochs` | `10` | Training epochs |
| `gradient_clip` | `1.0` | Gradient clipping norm |

---

## 🗺️ Roadmap & Future Integrations

- [x] **GPT-2** architecture
- [x] **Mistral** architecture with SWA + MoE
- [x] **LLaMA 2** architecture with RoPE
- [x] **LLaMA 3** architecture with GQA
- [ ] **Tokenizer Strategy Pattern** — pluggable tiktoken / SentencePiece / HuggingFace backends
- [ ] **Distributed Multi-GPU (DDP)** and FSDP training pipelines
- [ ] **Auxiliary Gating Load-Balancing Loss** for MoE stability
- [ ] **Learning Rate Schedules** (Cosine with Warmup) and Early Stopping
- [ ] **Gemma** architecture integration
- [ ] **Qwen 3** architecture integration

---

## 🤝 Contributors & Support

This project is created and maintained by **[Amzad Hossain Rafi](mailto:amzadhossain880@gmail.com)**.

If you are interested in collaborating, scaling up Bangla LLM capabilities, or have suggestions, feel free to **ping me or open an issue/PR!**

# 🇧🇩 BLLMC: Bangla LLM Collection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A highly modular, extensible, and high-performance PyTorch library for training and deploying **Bangla Language Models (LLMs)** from scratch. Built with a focus on modern LLM innovations including **Sliding Window Attention (SWA)**, **Rotary Position Embeddings (RoPE)**, and **Sparse Mixture of Experts (MoE)**.

> [!TIP]
> 🌟 **Don't forget to star the repository if you find it helpful!** Looking to collaborate? Feel free to reach out or open a Pull Request.

---

## ✨ Features

- 🏗️ **Modular & Composable Architecture**: Decoupled attention mechanisms, positional embeddings, normalizations, and feedforward blocks that can be easily plugged into any model.
- ⚡ **State-of-the-Art Core Layers**:
  - **Rotary Position Embeddings (RoPE)**: High-precision relative positional encoding.
  - **Sliding Window Attention (SWA)**: Reduced memory footprint for long-context modeling.
  - **Sparse Mixture of Experts (MoE)**: Gated sparse routing with parallel execution of top-k experts.
  - **Fused Operations**: Integrated with PyTorch's `scaled_dot_product_attention` for FlashAttention compatibility.
- 🏭 **Model Factory & Registry**: Easily instantiate architectures (`gpt2`, `mistral`, etc.) via a dynamic registration pattern.
- 📊 **Bangla-Optimized Data Pipeline**: Built-in corpus downloader, automated train/val/test splitter, and windowed token-by-token sequence generation loaders.
- 🚀 **Trainer Design Pattern**: A scalable LLM trainer supporting gradient clipping, validation monitoring, evaluation routines, and interactive text-generation prompts during training.

---

## 📁 Repository Structure

```directory
BLLMC/
├── src/BLLMC/                 # Core Source Package
│   ├── components/            # Neural Network Blocks & Modules
│   │   ├── attention/         # Attention Implementations
│   │   │   ├── sw_attention.py    # Sliding Window Attention with KV Caching & RoPE
│   │   │   ├── multi_head.py      # Standard Multi-Head Attention
│   │   │   ├── grouped_query.py   # Grouped-Query Attention (GQA)
│   │   │   └── multihead_latent.py # Multihead Latent Attention (MLA)
│   │   │
│   │   ├── layers/            # Neural Network Layers
│   │   │   ├── embeddings.py      # RoPE (Rotary Position Embeddings) calculation
│   │   │   ├── feedforward.py     # Sparse MoE FeedForward & Standard Dense FFN
│   │   │   ├── normalization.py   # LayerNorm & RMSNorm implementations
│   │   │   └── activations.py     # Custom Activations (GELU, SwiGLU)
│   │   │
│   │   ├── blocks/            # Unified Transformer Blocks
│   │   │   ├── gpt2_block.py      # Standard Attention + FFN Block
│   │   │   ├── mistral_block.py   # RMSNorm + SWA + MoE FeedForward Block
│   │   │   └── llama_block.py     # Llama-style Block
│   │   │
│   │   ├── config.py          # Unified Composable Dataclass Configs
│   │   ├── models.py          # Model Registry and Architecture Factories
│   │   └── trainer.py         # Abstract Trainer and LLMTrainer Implementation
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
├── artifacts/                 # Checkpoints, Model Weights, and Markdown Docs
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

### 4. Running the Training Loop

Train a model from scratch using the high-level train pipeline. You can switch model architectures (e.g. `gpt2` or `mistral` with MoE) by modifying your configuration:

```python
from BLLMC.components.config import mistral_config
from BLLMC.components.models import ModelFactory
from BLLMC.data.loader import create_dataloader
from BLLMC.components.trainer import LLMTrainer

# 1. Initialize custom Mistral-style config with sparse Mixture of Experts (MoE)
config = mistral_config(
    batch_size=8,
    max_epochs=5,
    num_experts=8,             # Total experts in MoE FFN
    num_experts_per_tok=2,     # Top-2 sparse routing
)

# 2. Build the model via the registry
model = ModelFactory.create_model(config)

# 3. Read ingestion splits
with open(config.train_data_path, "r", encoding="utf-8") as f:
    train_data = f.read()
with open(config.val_data_path, "r", encoding="utf-8") as f:
    val_data = f.read()

# 4. Create optimized DataLoaders
train_loader = create_dataloader(train_data, "gpt2", config)
val_loader = create_dataloader(val_data, "gpt2", config)

# 5. Kick off training
trainer = LLMTrainer(model, train_loader, val_loader, config)
trainer.train()
```

---

## 🧠 Supported Architectures

### 1. GPT-2 Style Model
- Dense FeedForward layers.
- Learned Positional Embeddings.
- Standard LayerNorm and Causal Multi-Head Attention.

### 2. Mistral Style Model
- **RoPE**: Rotary Position Embeddings for dynamic sequence extrapolation.
- **SWA**: Sliding Window Attention to handle longer context with constant memory footprints.
- **MoE**: Sparse Mixture of Experts with Gated Linear Unit (GLU) feedforward routing.

---

## 🗺️ Roadmap & Future Architecture Integrations
- [ ] **Gemma 3** architecture integration.
- [ ] **Gemma 4** architecture integration.
- [ ] **Qwen 3** architecture integration.
- [ ] **Distributed Multi-GPU (DDP)** and FSDP training pipelines.
- [ ] **Auxiliary Gating Load-Balancing Loss** for MoE stability.
- [ ] **Learning rate schedules (Cosine with Warmup)** and Early Stopping.

---

## 🤝 Contributors & Support

This project is created and maintained by **[Amzad Hossain Rafi](mailto:amzadhossain880@gmail.com)**. 

If you are interested in collaborating, scaling up Bangla LLM capabilities, or have suggestions, feel free to **ping me or open an issue/PR!**

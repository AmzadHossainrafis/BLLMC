import torch
import torch.nn as nn

# Importing configuration classes
from BLLMC.components.config import GPT_Config
from BLLMC.components.blocks.gpt2_block import TransformerBlock
from BLLMC.components.blocks.mistral_block import MistralBlock
from BLLMC.components.blocks.llama_block import Llama3Block, Llama2Block


class ModelFactory:
    """
    Factory class to instantiate models based on the provided configuration.
    Registers models dynamically to avoid a growing if-else chain.
    """

    _registry = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class with a specific architecture name."""

        def decorator(model_class):
            cls._registry[name.lower()] = model_class
            return model_class

        return decorator

    @classmethod
    def create_model(cls, config: GPT_Config) -> nn.Module:
        """
        Creates and returns a PyTorch model based on the architecture specified in the config.
        """
        architecture = config.architecture.lower()

        if architecture not in cls._registry:
            available = ", ".join(f"'{k}'" for k in cls._registry.keys())
            raise ValueError(
                f"Unsupported model architecture: '{architecture}'. "
                f"Available architectures are: {available}."
            )

        return cls._registry[architecture](config)


@ModelFactory.register("gpt2")
class GPT2Model(nn.Module):
    """
    GPT-2 model architecture.

    Architecture:
        Token Embedding → [TransformerBlock x n_layers] → LayerNorm → LM Head

    Each TransformerBlock contains:
        - MultiHeadAttention → Residual
        - FeedForward → Residual
    """
    def __init__(self, config: GPT_Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim)
        self.position_embeddings = nn.Embedding(config.context_length, config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)  # Fixed: drop_rate

        # Fixed: unpacked list for nn.Sequential
        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embeddings.weight

    def forward(self, in_idx):
        seq_len = in_idx.shape[1]  # Fixed: Dynamic sequence length
        token_emb = self.embeddings(in_idx)

        position_emb = self.position_embeddings(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = token_emb + position_emb
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)


@ModelFactory.register("mistral")
class MistralModel(nn.Module):
    """
    Mistral-style decoder model.

    Architecture:
        Token Embedding → [MistralBlock x n_layers] → RMSNorm → LM Head

    Each MistralBlock contains:
        - Pre-RMSNorm → SlidingWindowAttention → Residual
        - Pre-RMSNorm → MoEFeedForward → Residual

    default config from Mistral-7B:
        emb_dim = 4096
        n_heads = 32
        context_length = 8192
        drop_rate = 0.0
        vocab_size = 32000
    """

    def __init__(self, config: GPT_Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim)
        self.dropout = nn.Dropout(config.drop_rate)
        self.blocks = nn.ModuleList(
            [MistralBlock(config) for _ in range(config.n_layers)]
        )
        self.rms_norm_f = nn.RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx, use_cache=False):
        token_emb = self.embeddings(in_idx)
        x = self.dropout(token_emb)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.rms_norm_f(x)
        return self.lm_head(x)

    def reset_cache(self):
        for block in self.blocks:
            if hasattr(block, "reset_cache"):
                block.reset_cache()


@ModelFactory.register("llama3")
class Llama3Model(nn.Module):
    """
    Llama3-style decoder model.

    Architecture:
        Token Embedding → [Llama3Block x n_layers] → RMSNorm → LM Head

    Each Llama3Block contains:
        - Pre-RMSNorm → MultiHeadAttention → Residual
        - Pre-RMSNorm → FeedForward → Residual

    default config from Llama3-8B:
        emb_dim = 4096
        n_heads = 32
        context_length = 8192
        drop_rate = 0.0
        vocab_size = 128000
    """
    def __init__(self, config: GPT_Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim)
        self.blocks = nn.ModuleList(
            [Llama3Block(config) for _ in range(config.n_layers)]
        )
        self.rms_norm_f = nn.RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx, use_cache=False):
        x = self.embeddings(in_idx)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.rms_norm_f(x)
        return self.lm_head(x)

    def reset_cache(self):
        for block in self.blocks:
            if hasattr(block, "reset_cache"):
                block.reset_cache()

@ModelFactory.register("llama2")
class LlamaModel(nn.Module):
    """
    Llama2-style decoder model.

    Architecture:
        Token Embedding → [Llama2Block x n_layers] → RMSNorm → LM Head

    Each Llama2Block contains:
        - Pre-RMSNorm → MultiHeadAttentionWithRoPE → Residual
        - Pre-RMSNorm → FeedForward → Residual

    default config from Llama2-7B:
        emb_dim = 4096
        n_heads = 32
        context_length = 2048
        drop_rate = 0.0
        vocab_size = 32000
    """
    def __init__(self, config: GPT_Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.emb_dim)
        self.blocks = nn.ModuleList(
            [Llama2Block(config) for _ in range(config.n_layers)]
        )
        self.rms_norm_f = nn.RMSNorm(config.emb_dim)
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx, use_cache=False):
        token_emb = self.embeddings(in_idx)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        x = self.rms_norm_f(x)
        return self.lm_head(x)

    def reset_cache(self):
        for block in self.blocks:
            if hasattr(block, "reset_cache"):
                block.reset_cache()


@ModelFactory.register("Qwen3")
class Qwen3Model(nn.Module):
    def __init__(self, config: GPT_Config):
        super().__init__()
        self.config = config
        pass

    def forward(self, in_idx, use_cache=False):
        pass

    def reset_cache(self):
        pass


@ModelFactory.register("gemma4")
class Gemma4Model(nn.Module):
    def __init__(self, config: GPT_Config):
        super().__init__()
        self.config = config
        pass

    def forward(self, in_idx, use_cache=False):
        pass

    def reset_cache(self):
        pass
import torch
import torch.nn as nn
from typing import Union

# Importing configuration classes
from BLLMC.components.config import GPT_Config
from BLLMC.components.blocks.gpt2_block import TransformerBlock


class GPT2Model(nn.Module):
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


class LlamaModel(nn.Module):
    def __init__(self, config: GPT_Config):
        super().__init__()
        self.config = config
        # TODO: Initialize LLaMA specific components

    def forward(self, x):
        # TODO: Implement forward pass
        return x


class ModelFactory:
    """
    Factory class to instantiate models based on the provided configuration.
    """

    @staticmethod
    def create_model(config: GPT_Config) -> nn.Module:
        """
        Creates and returns a PyTorch model based on the architecture specified in the config.

        Args:
            config: A configuration object containing model hyper-parameters and architecture type.

        Returns:
            An instance of the requested PyTorch model.

        Raises:
            ValueError: If the architecture specified in the config is not supported.
        """
        # Ensure we're checking in a case-insensitive way
        architecture = config.architecture.lower()

        if architecture == "gpt2":
            return GPT2Model(config)
        elif architecture == "llama":
            return LlamaModel(config)
        else:
            raise ValueError(
                f"Unsupported model architecture: '{architecture}'. Available architectures are: 'gpt2', 'llama'."
            )

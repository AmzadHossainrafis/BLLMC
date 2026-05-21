"""
licence : mit
author : amzad hossain rafi
email : [EMAIL_ADDRESS]

change log :
    17-5-2026 : start
    17-5-2026 : implement multi head attention

to do :
    1. Implement test cases

"""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    This module implements the standard multi-head self-attention mechanism,
    projecting the input into multiple query, key, and value representations.
    It applies causal masking to prevent attending to future tokens.

    Args:
        config: A configuration object must contain:
            - emb_dim (int): Embedding dimension (used for both input and output projections).
            - n_heads (int): Number of attention heads. emb_dim must be divisible by n_heads.
            - context_length (int): Maximum sequence length for the causal mask.
            - drop_rate (float): Dropout probability applied to attention weights.
    """

    def __init__(self, config):
        """
        Initializes the MultiHeadAttention module.

        Args:
            config: A configuration object must contain:
                - emb_dim (int): Embedding dimension.
                - n_heads (int): Number of attention heads.
                - context_length (int): Context window size for masking.
                - drop_rate (float): Dropout rate.
        """
        super().__init__()
        assert config.emb_dim % config.n_heads == 0

        self.d_out = config.emb_dim
        self.n_head = config.n_heads
        self.head_dim = config.emb_dim // config.n_heads

        ## layere
        self.wq = nn.Linear(config.emb_dim, self.d_out, bias=False)
        self.wk = nn.Linear(config.emb_dim, self.d_out, bias=False)
        self.wv = nn.Linear(config.emb_dim, self.d_out, bias=False)

        self.out_proj = nn.Linear(self.d_out, self.d_out, bias=False)
        self.dropout = nn.Dropout(config.drop_rate)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(config.context_length, config.context_length), diagonal=1
            ).bool(),
        )


        #k cache 
        self.register_buffer("k_cache", None , persistent=False)
        self.register_buffer("v_cache", None , persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the multi-head attention for the given input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, num_tokens, emb_dim).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, num_tokens, emb_dim) after attention
                and final projection.
        """
        Batch, num_tokens, d_in = x.shape

        # Q K V projection
        q = self.wq(x)  # (B, T, d_out)
        k = self.wk(x)  # (B, T, d_out)
        v = self.wv(x)  # (B, T, d_out)

        # Split into multiple heads
        q = q.view(Batch, num_tokens, self.n_head, self.head_dim)
        k = k.view(Batch, num_tokens, self.n_head, self.head_dim)
        v = v.view(Batch, num_tokens, self.n_head, self.head_dim)

        # transpose into (B,n_head,T,head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attention_scores = q @ k.transpose(2, 3)

        mask = self.mask[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill_(mask, -torch.inf)
        attention_weights = torch.softmax(attention_scores / k.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = (attention_weights @ v).transpose(1, 2).contiguous()
        context = context.view(Batch, num_tokens, self.d_out)

        return self.out_proj(context)


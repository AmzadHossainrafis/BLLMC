"""
License: MIT
Author: Amzad Hossain Rafi
Date: 2026-06-11
"""

from abc import ABC, abstractmethod
import os
from typing import List

import sentencepiece as spm
import tiktoken
from huggingface_hub import hf_hub_download


class TokenizerStrategy(ABC):
    """Abstract base class representing a tokenizer strategy."""

    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode a string of text into a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode a list of token IDs back into a string of text."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        pass


class TiktokenStrategy(TokenizerStrategy):
    """Tokenizer strategy using OpenAI's tiktoken library."""

    def __init__(self, encoding_name: str = "gpt2"):
        self._tokenizer = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str, **kwargs) -> List[int]:
        allowed_special = kwargs.get("allowed_special", "all")
        return self._tokenizer.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.n_vocab


class SentencePieceStrategy(TokenizerStrategy):
    """Tokenizer strategy using Google's SentencePiece library."""

    def __init__(self, model_path: str):
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.Load(model_path)

    def encode(self, text: str, **kwargs) -> List[int]:
        return self._tokenizer.Encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._tokenizer.Decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.GetPieceSize()


def get_tokenizer(config) -> TokenizerStrategy:
    """Factory function to instantiate the correct TokenizerStrategy from the configuration.

    If the specified SentencePiece model path does not exist locally,
    this function attempts to download the model file from Hugging Face Hub.
    """
    backend = config.tokenizer_backend.lower()

    if backend == "tiktoken":
        return TiktokenStrategy(config.tokenizer_model)

    elif backend == "sentencepiece":
        model_path = config.tokenizer_model

        if not os.path.exists(model_path):
            repo_id = getattr(config, "repo_id", None)
            filename = getattr(config, "filename", None)

            if repo_id and filename:
                try:
                    local_dir = getattr(
                        config, "local_path", "./dataset/tokenizer_model"
                    )
                    token = getattr(config, "hf_token", None)

                    print(
                        f"Downloading tokenizer model from Hugging Face ({repo_id})..."
                    )
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=local_dir,
                        token=token,
                    )
                    model_path = downloaded_path
                except Exception as e:
                    raise FileNotFoundError(
                        f"Could not load local tokenizer from '{model_path}', and auto-download failed: {e}"
                    )
            else:
                raise FileNotFoundError(
                    f"Tokenizer model not found at '{model_path}' and no Hugging Face configuration was provided."
                )

        return SentencePieceStrategy(model_path)

    else:
        raise ValueError(f"Unknown tokenizer backend: {config.tokenizer_backend}")

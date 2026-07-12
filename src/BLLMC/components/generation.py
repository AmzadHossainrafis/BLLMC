from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from BLLMC.components.config import GPT_Config
from BLLMC.utils.logger import logger


class GenerationStrategy(ABC):
    """
    Strategy interface for text generation.
    Each concrete strategy implements a different sampling method
    but shares the same generate() interface.
    """

    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        idx: torch.Tensor,
        max_new_tokens: int,
        context_size: int,
        device_type: str,
        amp_dtype: torch.dtype,
        use_amp: bool,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        pass


class GreedyGenerationStrategy(GenerationStrategy):
    """Greedy text generation strategy (always selects
    the highest-probability token)."""

    def generate(
        self,
        model: nn.Module,
        idx: torch.Tensor,
        max_new_tokens: int,
        context_size: int,
        device_type: str,
        amp_dtype: torch.dtype,
        use_amp: bool,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device_type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    logits = model(idx_cond)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_id is not None and (idx_next == eos_id).all():
                break
            idx = torch.cat((idx, idx_next), dim=1)
        model.train()
        return idx


class TemperatureGenerationStrategy(GenerationStrategy):
    """Temperature-based sampling strategy."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def generate(
        self,
        model: nn.Module,
        idx: torch.Tensor,
        max_new_tokens: int,
        context_size: int,
        device_type: str,
        amp_dtype: torch.dtype,
        use_amp: bool,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device_type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    logits = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits / self.temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if eos_id is not None and (idx_next == eos_id).all():
                break
            idx = torch.cat((idx, idx_next), dim=1)
        model.train()
        return idx


class TopKGenerationStrategy(GenerationStrategy):
    """Top-K sampling strategy (with optional temperature)."""

    def __init__(self, top_k: int = 50, temperature: float = 1.0):
        self.top_k = top_k
        self.temperature = temperature

    def generate(
        self,
        model: nn.Module,
        idx: torch.Tensor,
        max_new_tokens: int,
        context_size: int,
        device_type: str,
        amp_dtype: torch.dtype,
        use_amp: bool,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type=device_type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    logits = model(idx_cond)
            logits = logits[:, -1, :]
            if self.temperature != 1.0:
                logits = logits / self.temperature

            # Filter top-k
            v, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if eos_id is not None and (idx_next == eos_id).all():
                break
            idx = torch.cat((idx, idx_next), dim=1)
        model.train()
        return idx


def get_generation_strategy(config: GPT_Config) -> GenerationStrategy:
    """Helper factory to retrieve the configured text generation strategy."""
    strategy_name = getattr(config, "gen_strategy", "greedy").lower()
    temperature = getattr(config, "gen_temperature", 1.0)
    top_k = getattr(config, "gen_top_k", 50)

    if strategy_name == "greedy" or strategy_name == "generation":
        return GreedyGenerationStrategy()
    elif strategy_name == "temperature" or strategy_name == "temp_generation":
        return TemperatureGenerationStrategy(temperature=temperature)
    elif strategy_name == "top_k" or strategy_name == "top_k_generation":
        return TopKGenerationStrategy(top_k=top_k, temperature=temperature)
    else:
        logger.warning(
            f"Unknown generation strategy: {strategy_name}. Falling back to Greedy."
        )
        return GreedyGenerationStrategy()

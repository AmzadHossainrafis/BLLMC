import torch
import torch.nn as nn
from BLLMC.components.config import GPT_Config
from BLLMC.components import (
    GreedyGenerationStrategy,
    TemperatureGenerationStrategy,
    TopKGenerationStrategy,
    get_generation_strategy,
)


class MockModel(nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, idx):
        # Return dummy logits of shape (batch_size, sequence_length, vocab_size)
        batch_size, seq_len = idx.shape
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=idx.device)
        # Make the last token of vocabulary highly likely, except for the last step
        logits[:, :, -1] = 10.0
        return logits


def test_greedy_generation_strategy():
    model = MockModel(vocab_size=10)
    strategy = GreedyGenerationStrategy()
    idx = torch.tensor([[1, 2, 3]])

    # Generate 3 new tokens. Since it's greedy, it should always choose the argmax token (index 9)
    result = strategy.generate(
        model=model,
        idx=idx,
        max_new_tokens=3,
        context_size=5,
        device_type="cpu",
        amp_dtype=torch.float32,
        use_amp=False,
    )

    assert result.shape == (1, 6)
    assert result[0, 3].item() == 9
    assert result[0, 4].item() == 9
    assert result[0, 5].item() == 9


def test_temperature_generation_strategy():
    model = MockModel(vocab_size=10)
    strategy = TemperatureGenerationStrategy(temperature=0.5)
    idx = torch.tensor([[1, 2, 3]])

    result = strategy.generate(
        model=model,
        idx=idx,
        max_new_tokens=2,
        context_size=5,
        device_type="cpu",
        amp_dtype=torch.float32,
        use_amp=False,
    )

    assert result.shape == (1, 5)


def test_top_k_generation_strategy():
    model = MockModel(vocab_size=10)
    strategy = TopKGenerationStrategy(top_k=2, temperature=1.0)
    idx = torch.tensor([[1, 2, 3]])

    result = strategy.generate(
        model=model,
        idx=idx,
        max_new_tokens=2,
        context_size=5,
        device_type="cpu",
        amp_dtype=torch.float32,
        use_amp=False,
    )

    assert result.shape == (1, 5)


def test_get_generation_strategy():
    # Test greedy strategy
    config = GPT_Config()
    config.gen_strategy = "greedy"
    strategy = get_generation_strategy(config)
    assert isinstance(strategy, GreedyGenerationStrategy)

    # Test temperature strategy
    config.gen_strategy = "temperature"
    config.gen_temperature = 0.8
    strategy = get_generation_strategy(config)
    assert isinstance(strategy, TemperatureGenerationStrategy)
    assert strategy.temperature == 0.8

    # Test top-k strategy
    config.gen_strategy = "top_k"
    config.gen_top_k = 10
    config.gen_temperature = 0.5
    strategy = get_generation_strategy(config)
    assert isinstance(strategy, TopKGenerationStrategy)
    assert strategy.top_k == 10
    assert strategy.temperature == 0.5

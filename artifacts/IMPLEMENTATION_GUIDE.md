# Implementation Guide: From Design to Code

## PART 1: BASE CLASSES & INTERFACES

### Step 1: Create Base Model Class
**File**: `src/bllmc/models/base/base_model.py`

```python
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import json
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models in BLLMC.
    All models should inherit from this class.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_training = True
    
    @abstractmethod
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_input_embeddings(self):
        """Return the embeddings layer"""
        pass
    
    @abstractmethod
    def set_input_embeddings(self, new_embeddings):
        """Set new embeddings layer"""
        pass
    
    def save_pretrained(self, save_directory):
        """
        Save model checkpoint and config
        
        Args:
            save_directory: Path to save checkpoint
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # Save config
        with open(save_path / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """
        Load model from checkpoint
        
        Args:
            model_name_or_path: Path to model checkpoint or model name
            **kwargs: Additional arguments
            
        Returns:
            Loaded model instance
        """
        from bllmc.registry import ConfigRegistry
        
        model_path = Path(model_name_or_path)
        
        # Load config
        config_path = model_path / "config.json" if model_path.is_dir() else None
        if not config_path or not config_path.exists():
            # Try loading from registry
            config = ConfigRegistry.get(model_name_or_path)
        else:
            with open(config_path) as f:
                config_dict = json.load(f)
                # Convert dict to config object
                from bllmc.config import ModelConfig
                config = ModelConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        if model_path.is_dir():
            weights_path = model_path / "pytorch_model.bin"
            if weights_path.exists():
                state_dict = torch.load(weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
        
        return model
    
    def train(self, mode=True):
        """Set training mode"""
        self._is_training = mode
        return super().train(mode)
    
    def eval(self):
        """Set evaluation mode"""
        self._is_training = False
        return super().eval()


# ----- BERT-STYLE OUTPUT -----

class BertOutput:
    """Output container for BERT models"""
    def __init__(self, last_hidden_state, pooled_output=None, hidden_states=None, 
                 attentions=None):
        self.last_hidden_state = last_hidden_state
        self.pooled_output = pooled_output
        self.hidden_states = hidden_states
        self.attentions = attentions
    
    def __getitem__(self, index):
        if index == 0:
            return self.last_hidden_state
        elif index == 1:
            return self.pooled_output
        else:
            raise IndexError(f"Invalid index: {index}")
```

### Step 2: Create Registry System
**File**: `src/bllmc/registry.py`

```python
from typing import Dict, Type, Any
import json
from pathlib import Path


class Registry:
    """Generic registry for models, tokenizers, and configs"""
    
    def __init__(self):
        self._registry: Dict[str, Any] = {}
    
    def register(self, name: str):
        """Decorator to register items"""
        def decorator(item):
            if name in self._registry:
                print(f"Overwriting existing entry for '{name}'")
            self._registry[name] = item
            return item
        return decorator
    
    def get(self, name: str):
        """Retrieve item from registry"""
        if name not in self._registry:
            raise ValueError(
                f"'{name}' not found. Available: {list(self._registry.keys())}"
            )
        return self._registry[name]
    
    def list_available(self):
        """List all available items"""
        return list(self._registry.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._registry
    
    def __len__(self) -> int:
        return len(self._registry)


# Global registries
MODEL_REGISTRY = Registry()
TOKENIZER_REGISTRY = Registry()
CONFIG_REGISTRY = Registry()


# Convenience decorators
def register_model(name: str):
    """Register a model class"""
    return MODEL_REGISTRY.register(name)


def register_tokenizer(name: str):
    """Register a tokenizer class"""
    return TOKENIZER_REGISTRY.register(name)


def register_config(name: str):
    """Register a configuration"""
    return CONFIG_REGISTRY.register(name)


# Functions to load configurations
def load_config(config_path: str) -> Dict:
    """Load config from JSON file"""
    with open(config_path) as f:
        return json.load(f)


def save_config(config: Dict, save_path: str):
    """Save config to JSON file"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
```

### Step 3: Create Configuration Classes
**File**: `src/bllmc/config/model_config.py`

```python
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    
    # Vocabulary
    vocab_size: int = 50000
    
    # Model dimensions
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    
    # Activation and dropout
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Position and layer norm
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    
    # Model type
    model_type: str = "bert"  # 'bert', 'gpt', 't5'
    
    # Type vocab
    type_vocab_size: int = 2
    
    # Gradient checkpointing
    gradient_checkpointing: bool = False
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary"""
        return cls(**config_dict)
    
    def __repr__(self):
        return f"ModelConfig({self.to_dict()})"


@dataclass
class TokenizerConfig:
    """Configuration for tokenizers"""
    
    vocab_size: int = 50000
    max_length: int = 512
    padding: str = "max_length"  # 'max_length', 'longest'
    truncation: bool = True
    special_tokens: dict = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                'pad_token': '[PAD]',
                'unk_token': '[UNK]',
                'cls_token': '[CLS]',
                'sep_token': '[SEP]',
                'mask_token': '[MASK]',
            }
```

### Step 4: Create Factory Pattern
**File**: `src/bllmc/models/factory.py`

```python
from typing import Optional, Dict, Any
from bllmc.registry import MODEL_REGISTRY, CONFIG_REGISTRY
from bllmc.config import ModelConfig


class ModelFactory:
    """Factory for creating models"""
    
    @staticmethod
    def create_model(model_type: str, config: Optional[ModelConfig] = None, 
                     **kwargs) -> 'BaseModel':
        """
        Create a model instance
        
        Args:
            model_type: Type of model ('bert', 'gpt', etc.)
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        if config is None:
            config = ModelConfig(model_type=model_type)
        
        ModelClass = MODEL_REGISTRY.get(model_type)
        return ModelClass(config, **kwargs)
    
    @staticmethod
    def from_pretrained(model_name: str, **kwargs) -> 'BaseModel':
        """
        Load a pre-trained model
        
        Args:
            model_name: Model name (e.g., 'bllmc/bengali-bert-base')
            **kwargs: Additional arguments
            
        Returns:
            Loaded model instance
        """
        ModelClass = MODEL_REGISTRY.get(model_name)
        return ModelClass.from_pretrained(model_name, **kwargs)


class TokenizerFactory:
    """Factory for creating tokenizers"""
    
    @staticmethod
    def create_tokenizer(tokenizer_type: str, vocab_path: str, **kwargs):
        """Create tokenizer instance"""
        from bllmc.registry import TOKENIZER_REGISTRY
        TokenizerClass = TOKENIZER_REGISTRY.get(tokenizer_type)
        return TokenizerClass(vocab_path=vocab_path, **kwargs)
    
    @staticmethod
    def from_pretrained(tokenizer_name: str, **kwargs):
        """Load pre-trained tokenizer"""
        from bllmc.registry import TOKENIZER_REGISTRY
        TokenizerClass = TOKENIZER_REGISTRY.get(tokenizer_name)
        return TokenizerClass.from_pretrained(tokenizer_name, **kwargs)
```

### Step 5: Create Auto API
**File**: `src/bllmc/auto/auto_model.py`

```python
from typing import Optional
from bllmc.registry import MODEL_REGISTRY, CONFIG_REGISTRY
from bllmc.config import ModelConfig


class AutoModel:
    """Automatically load models like HuggingFace Transformers"""
    
    @staticmethod
    def from_pretrained(model_name: str, **kwargs):
        """
        Load a pre-trained model by name
        
        Example:
            model = AutoModel.from_pretrained('bllmc/bengali-bert-base')
        """
        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {MODEL_REGISTRY.list_available()}"
            )
        
        ModelClass = MODEL_REGISTRY.get(model_name)
        return ModelClass.from_pretrained(model_name, **kwargs)
    
    @staticmethod
    def from_config(config: ModelConfig, **kwargs):
        """Create model from config"""
        if config.model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        ModelClass = MODEL_REGISTRY.get(config.model_type)
        return ModelClass(config, **kwargs)


class AutoTokenizer:
    """Automatically load tokenizers"""
    
    @staticmethod
    def from_pretrained(tokenizer_name: str, **kwargs):
        """
        Load a pre-trained tokenizer by name
        
        Example:
            tokenizer = AutoTokenizer.from_pretrained('bllmc/bengali-bert-base')
        """
        from bllmc.registry import TOKENIZER_REGISTRY
        
        if tokenizer_name not in TOKENIZER_REGISTRY:
            raise ValueError(
                f"Tokenizer '{tokenizer_name}' not found. "
                f"Available: {TOKENIZER_REGISTRY.list_available()}"
            )
        
        TokenizerClass = TOKENIZER_REGISTRY.get(tokenizer_name)
        return TokenizerClass.from_pretrained(tokenizer_name, **kwargs)


class AutoConfig:
    """Automatically load configurations"""
    
    @staticmethod
    def from_pretrained(config_name: str):
        """Load a pre-trained config"""
        if config_name not in CONFIG_REGISTRY:
            raise ValueError(f"Config '{config_name}' not found")
        
        config = CONFIG_REGISTRY.get(config_name)
        return config
```

---

## PART 2: COMPONENT IMPLEMENTATIONS

### Step 6: Create Attention Mechanism (Strategy Pattern)
**File**: `src/bllmc/components/attention/multi_head.py`

```python
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class AttentionBase(nn.Module, ABC):
    """Abstract base for attention mechanisms"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, 
                 attention_dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )
        
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.scale_factor = self.head_dim ** -0.5
    
    @abstractmethod
    def forward(self, query, key, value, attention_mask=None):
        pass


class MultiHeadAttention(AttentionBase):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, hidden_size: int, num_attention_heads: int, 
                 attention_dropout: float = 0.1):
        super().__init__(hidden_size, num_attention_heads, attention_dropout)
        
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, key, value, attention_mask=None):
        """
        Args:
            query: (batch_size, seq_len_q, hidden_size)
            key: (batch_size, seq_len_k, hidden_size)
            value: (batch_size, seq_len_v, hidden_size)
            attention_mask: (batch_size, seq_len_q, seq_len_k)
        
        Returns:
            output: (batch_size, seq_len_q, hidden_size)
        """
        batch_size = query.size(0)
        
        # Project
        Q = self.query_projection(query)
        K = self.key_projection(key)
        V = self.value_projection(value)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len_q, head_dim)
        
        K = K.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        K = K.transpose(1, 2)  # (batch, num_heads, seq_len_k, head_dim)
        
        V = V.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        V = V.transpose(1, 2)  # (batch, num_heads, seq_len_v, head_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale_factor
        
        # Apply mask
        if attention_mask is not None:
            # Expand mask for multi-head
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Context
        context = torch.matmul(attn_weights, V)
        
        # Combine heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        # Output projection
        output = self.output_projection(context)
        
        return output
```

---

## PART 3: QUICK START TEMPLATE

### Complete Minimal Example
**File**: `examples/1_basic_usage.py`

```python
"""
Basic usage example for BLLMC
"""

import torch
from bllmc.auto import AutoModel, AutoTokenizer
from bllmc.config import ModelConfig
from bllmc.registry import register_model


# Step 1: Create a simple BERT model
@register_model('simple-bengali-bert')
class SimpleBengaliBERT:
    pass  # Implementation coming in next sections


# Step 2: Load components
if __name__ == "__main__":
    # Create a custom config
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        model_type='bert'
    )
    
    print(f"Config created: {config}")
    
    # Later: Load pre-trained
    # model = AutoModel.from_pretrained('bllmc/bengali-bert-base')
    # tokenizer = AutoTokenizer.from_pretrained('bllmc/bengali-bert-base')
```

---

## PART 4: TESTING TEMPLATE

**File**: `tests/test_base.py`

```python
import unittest
import torch
from bllmc.config import ModelConfig
from bllmc.components.attention import MultiHeadAttention


class TestMultiHeadAttention(unittest.TestCase):
    
    def setUp(self):
        self.hidden_size = 768
        self.num_heads = 12
        self.seq_len = 512
        self.batch_size = 4
        
        self.attention = MultiHeadAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_heads
        )
    
    def test_forward_pass(self):
        Q = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        K = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        V = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        output = self.attention(Q, K, V)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))
    
    def test_with_attention_mask(self):
        Q = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        K = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        V = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        # Create causal mask
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        output = self.attention(Q, K, V, attention_mask=mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_size))


if __name__ == '__main__':
    unittest.main()
```

---

## Next: Implementing Full Models

After implementing these base classes, proceed with:

1. **Embeddings Layer** - Word, Position, Token-Type embeddings
2. **Transformer Block** - Combines attention + feed-forward
3. **Full Models** - BERT, GPT, T5 architectures
4. **Tokenizers** - Bengali-specific tokenization
5. **Training Pipeline** - Trainer, loss functions, callbacks

Each step builds on the previous, following the modular design pattern.


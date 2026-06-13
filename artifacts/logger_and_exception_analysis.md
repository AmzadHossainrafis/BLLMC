# BLLMC Logging & Exception Handling Analysis

This document provides a deep architectural analysis of where and how to introduce structured logging and custom exception handling in the BLLMC repository. 

Currently, the codebase uses a single generic `CustomException` wrapping system exceptions and a combination of standard `print()` statements and basic logging. Implementing domain-specific exceptions and structured logging will drastically improve diagnostic capabilities, library robustness, and training observability.

---

## 1. Exception Hierarchy Design

Instead of relying on standard built-in Python exceptions or a single `CustomException`, BLLMC should define a clear hierarchy of domain-specific exceptions. This allows users and downstream scripts to catch specific classes of errors (e.g., catching configuration errors separately from training exceptions).

### Proposed Hierarchy (`src/BLLMC/utils/exception.py`)

```python
class BLLMCError(Exception):
    """Base exception class for all BLLMC errors."""
    pass

class ConfigurationError(BLLMCError):
    """Raised when the model, data, or training configuration is invalid or missing required parameters."""
    pass

class DatasetError(BLLMCError):
    """Raised when raw datasets are missing, corrupted, or when train/val splits fail."""
    pass

class TokenizerError(BLLMCError):
    """Raised when tokenizers fail to initialize, download, encode, or decode text."""
    pass

class ModelArchitectureError(BLLMCError):
    """Raised when an unsupported model architecture is requested or key layers mismatch."""
    pass

class TrainingExecutionError(BLLMCError):
    """Raised when errors occur during training loops, validation, or optimization steps."""
    pass

class CheckpointError(BLLMCError):
    """Raised when model checkpoints fail to save or load (e.g., file corruption)."""
    pass

class DeviceAllocationError(BLLMCError):
    """Raised when cuda/device allocation fails or runs Out-of-Memory (OOM)."""
    pass
```

---

## 2. Module-by-Module Integration Blueprint

Below is the detailed mapping of the codebase, identifying the precise points where logger calls and custom exceptions should be integrated.

| Module / File | Target Location | Recommended Action | Exception / Log Details |
| :--- | :--- | :--- | :--- |
| **`components/config.py`** | `ModelConfig.__post_init__` | Replace `assert` statements with `ConfigurationError` checks. | Raise `ConfigurationError` when `emb_dim % n_heads != 0` or `n_heads % n_kv_heads != 0`. |
| | Module level | Add validation logs on startup. | `logger.debug("Validating ModelConfig: emb_dim=%d, n_heads=%d", ...)` |
| **`data/dataset.py`** | `download_dataset` | Replace standard generic try-except blocks. | Log `logger.info("Dataset already exists")` or `logger.warning(...)`. Raise `DatasetError` if `gdown` download fails. |
| **`data/ingestion.py`** | `initiate_data_ingestion` | Validate parameters before ingestion. | Raise `DatasetError` if `train_split + val_split + test_split != 1.0` or if raw data file is empty. |
| | Exception block | Wrap file reading errors. | Catch `FileNotFoundError` and raise `DatasetError(f"Dataset file not found at {path}")`. |
| | Success path | Log details about splits. | `logger.info("Ingestion completed: Train=%d, Val=%d, Test=%d chars", len(train_text), len(val_text), len(test_text))` |
| **`data/loader.py`** | `BanglaDataset.__init__` | Add validation checking. | Raise `DatasetError` if input `text` is empty or if max_length is invalid. |
| | `create_dataloader` | Replace try-except fallback prints. | Replace raw `try-except` pass with warning log: `logger.warning("Tiktoken failed; falling back to default tokenizer backend: %s", config.tokenizer_backend)`. |
| **`data/tokenizer.py`** | `get_tokenizer` | Replace all `print()` statements and custom built-in raises. | 1. Replace `print(...)` with `logger.info("Downloading tokenizer model...")`. <br>2. Raise `TokenizerError` instead of standard `FileNotFoundError` and `ValueError` for cleaner context tracking. |
| **`components/base.py`** | `Trainer._setup_optimizer` | Raise custom optimizer exception. | Raise `ModelArchitectureError(f"Optimizer {self.config.optimizer} is not registered or supported.")` |
| | `Trainer._save_checkpoint` | Improve error logging & typing. | Log details before saving. Catch exceptions and raise `CheckpointError`. |
| | `Trainer._load_checkpoint` | Gracefully handle corrupt files. | Catch exceptions (e.g., pickle load errors) and raise `CheckpointError`. |
| | `ModelFactory.create_model` | Raise registry mismatch error. | Raise `ModelArchitectureError` instead of `ValueError` when requested model architecture isn't registered. |
| **`components/trainer.py`**| `LLMTrainer.train` | Resolve indentation bug and wrap step errors. | 1. **Bug Fix**: Indent the batch processing loop inside the epoch loop.<br>2. Log periodic updates using `logger.info` instead of mixing `print` and `logger`. |
| | `LLMTrainer.train_step` | Catch OOM and scaling failures. | Catch `RuntimeError` matching "out of memory" patterns, free cache, log custom diagnostic details, and raise `DeviceAllocationError`. |
| | `LLMTrainer.generate` | Log generation parameters. | Log generation status: `logger.debug("Generating sequence with context length %d and max new tokens %d", context_size, max_new_tokens)` |

---

## 3. Concrete Code Examples

### A. Configuration Validation (`src/BLLMC/components/config.py`)

#### Current:
```python
assert (
    self.emb_dim % self.n_heads == 0
), f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
```

#### Proposed:
```python
from BLLMC.utils.exception import ConfigurationError
from BLLMC.utils.logger import logger

if self.emb_dim % self.n_heads != 0:
    logger.error("Configuration validation failed: emb_dim is not divisible by n_heads.")
    raise ConfigurationError(
        f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
    )
```

---

### B. Tokenizer Error Handling & Logging (`src/BLLMC/data/tokenizer.py`)

#### Current:
```python
print(f"Downloading tokenizer model from Hugging Face ({repo_id})...")
...
except Exception as e:
    raise FileNotFoundError(
        f"Could not load local tokenizer from '{model_path}', and auto-download failed: {e}"
    )
```

#### Proposed:
```python
from BLLMC.utils.exception import TokenizerError
from BLLMC.utils.logger import logger

logger.info("Downloading tokenizer model from Hugging Face repository: %s", repo_id)
...
except Exception as e:
    logger.error("Failed to fetch or load the tokenizer model from Hugging Face: %s", str(e))
    raise TokenizerError(
        f"Could not load local tokenizer from '{model_path}', and auto-download failed: {e}"
    ) from e
```

---

### C. CUDA OOM & Memory Logging (`src/BLLMC/components/trainer.py`)

When training LLMs, out-of-memory (OOM) errors are frequent. Intercepting memory exceptions, freeing cache, and logging GPU allocation details is crucial.

#### Proposed structure inside `train_step` / `train`:
```python
from BLLMC.utils.exception import DeviceAllocationError
from BLLMC.utils.logger import logger
import gc

try:
    # Forward & Backward Pass
    loss = self.train_step(inputs, targets)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        # Free memory immediately
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            logger.error("CUDA OOM encountered. Memory allocated: %.2f GB, Reserved: %.2f GB", allocated, reserved)
        raise DeviceAllocationError("CUDA Out of Memory during training step.") from e
    else:
        raise TrainingExecutionError(f"Unexpected runtime error during training: {e}") from e
```

---

## 4. Key Advantages of the Proposed System

1. **Failure Containment**: Domain-specific exceptions prevent errors in dataset downloads or model initialization from raising ambiguous built-in exceptions (`ValueError`/`KeyError`), which can be difficult to catch reliably.
2. **Observability**: Consistent log formats make it easy to integrate with logging aggregators or write logs to disk during hours/days of training runs.
3. **Graceful CUDA Recoveries**: Diagnostic hooks for CUDA memory tracking ensure that developers know exactly how much memory was occupied during a crash, rather than just seeing a generic `RuntimeError`.

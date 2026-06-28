from abc import ABC, abstractmethod
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from BLLMC.components.config import GPT_Config
from BLLMC.utils.logger import logger
from BLLMC.utils.exception import CustomException
import sys


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup — industry standard for LLM pretraining.

    Used by nanoGPT, LLaMA, GPT-NeoX, and most modern LLM training runs.
    Linear warmup for the first `warmup_steps`, then cosine decay to `min_lr`.
    """

    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max(max_steps, 1)
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        """Advance the scheduler by one optimizer step and update the learning rate."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        """Compute the learning rate for the current step."""
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / max(self.warmup_steps, 1))
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                self.max_steps - self.warmup_steps, 1
            )
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

class Trainer(ABC):
    """
    Encapsulates the training loop, evaluation, and checkpointing for the language model.
    Follows the Trainer Design Pattern.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: GPT_Config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = nn.CrossEntropyLoss()

        # Setup Automatic Mixed Precision (AMP)
        self._setup_amp()

        if hasattr(self.config, "compile") and self.config.compile:
            logger.info("Compiling model...")
            self.model = torch.compile(self.model)
            logger.info("Model compiled successfully")

    def _setup_amp(self):
        """Configure AMP settings once during initialization."""
        logger.info("Setting up AMP...")
        if "cuda" in self.device:
            self.device_type = "cuda"
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # bfloat16 does not need loss scaling
                logger.info("bfloat16 AMP enabled")
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
                logger.info("float16 AMP enabled with GradScaler")
            self.use_amp = True
            logger.info("AMP enabled")
        else:
            # CPU: AMP autocast only supports bfloat16, and benefits are minimal
            logger.warning("CPU detected, AMP not enabled")
            self.device_type = "cpu"
            self.amp_dtype = torch.bfloat16
            self.scaler = None
            self.use_amp = False

    def _setup_optimizer(self):
        if self.config.optimizer == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"Optimizer {self.config.optimizer} not supported."
            )

    def _setup_scheduler(self):
        """Cosine annealing with linear warmup — industry standard for LLM pretraining.

        Computes max_steps from the train_loader length, accounting for
        gradient accumulation so the schedule tracks optimizer steps, not
        micro-batch steps.
        """
        accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        steps_per_epoch = len(self.train_loader) // accum_steps
        max_steps = steps_per_epoch * self.config.max_epochs
        logger.info(
            f"LR Scheduler: cosine warmup | warmup={self.config.warmup_steps} | "
            f"max_steps={max_steps} | min_lr={self.config.min_lr}"
        )
        return CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            max_steps=max_steps,
            min_lr=self.config.min_lr,
        )

    def _save_checkpoint(self, epoch: int, loss: float):
        try:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir, f"ckpt_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error in saving checkpoint: {e}")
            raise CustomException(e, sys)

    def _load_checkpoint(self, checkpoint_path: str):
        """
        Safely loads a checkpoint file and restores model/optimizer state.

        Uses weights_only=True to prevent arbitrary code execution from
        untrusted checkpoint files (pickle deserialization vulnerability).

        Args:
            checkpoint_path (str): Path to the .pt checkpoint file.

        Returns:
            dict: The checkpoint dictionary containing 'epoch', 'loss',
                  'model_state_dict', and 'optimizer_state_dict'.
        """
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(
                f"Resumed from epoch {checkpoint['epoch']} "
                f"(loss: {checkpoint['loss']:.4f})"
            )
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            raise CustomException(e, sys)

    @torch.no_grad()
    def generate(self, prompt: str):
        pass

    @abstractmethod
    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        pass

    @abstractmethod
    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.model.eval()

    @abstractmethod
    def train(self):
        pass


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
            logger.info(f"Model {name} registered successfully")
            return model_class

        return decorator

    @classmethod
    def create_model(cls, config: GPT_Config) -> nn.Module:
        """
        Creates and returns a PyTorch model based on the architecture specified in the config.
        """

        logger.info("Creating model...")
        architecture = config.architecture.lower()
        if architecture not in cls._registry:
            available = ", ".join(f"'{k}'" for k in cls._registry.keys())
            raise ValueError(
                f"Unsupported model architecture: '{architecture}'. "
                f"Available architectures are: {available}."
            )
        logger.info(f"Creating model with architecture: {architecture}")
        return cls._registry[architecture](config)


class TextGeneratorStrategy(ABC):
    @abstractmethod
    def generate(self, prompt: str):
        pass

    @abstractmethod
    def temp_generate(self, prompt: str, temperature: float = 1.0):
        pass

    def top_p_generate(self, prompt: str, top_p: float = 0.9):
        pass

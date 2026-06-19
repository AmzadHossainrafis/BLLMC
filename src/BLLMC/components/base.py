from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from BLLMC.components.config import GPT_Config
from BLLMC.utils.logger import logger
from BLLMC.utils.exception import CustomException
import sys


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

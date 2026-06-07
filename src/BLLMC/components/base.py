from abc import ABC, abstractmethod


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
            print("Compiling model...")
            self.model = torch.compile(self.model)

    def _setup_amp(self):
        """Configure AMP settings once during initialization."""
        if "cuda" in self.device:
            self.device_type = "cuda"
            if torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                self.scaler = None  # bfloat16 does not need loss scaling
            else:
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
            self.use_amp = True
        else:
            # CPU: AMP autocast only supports bfloat16, and benefits are minimal
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
        print(f"Saved checkpoint to {checkpoint_path}")

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

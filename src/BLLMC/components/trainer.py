"""
licence : mit
author : amzad hossain rafi
email : amzad.rafi@northsouth.edu

change log :
    17-5-2026 : start
    23-5-2026 : implement trainer design pattern
    8-6-2026 : implement Amp traing loop



#TODO :
    1. add learning rate scheduler
    2. add early stopping
    3. add distributed multi-gpu training

"""

import os
import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import DataLoader
from BLLMC.components.config import GPT_Config
from BLLMC.components.base import Trainer
from BLLMC.utils.logger import logger
from BLLMC.utils.exception import CustomException


class LLMTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: GPT_Config,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(model, train_loader, val_loader, config, device)

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor):
        self.optimizer.zero_grad()

        # Forward pass with AMP autocast
        with torch.amp.autocast(
            device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp
        ):
            model_output = self.model(inputs)
            loss = self.criterion(
                model_output.view(-1, model_output.size(-1)), targets.view(-1)
            )

        # Backward pass with optional loss scaling (float16 only)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader):
        self.model.eval()
        val_losses = []

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with torch.amp.autocast(
                device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1)
                )
            val_losses.append(loss.item())

        self.model.train()
        return torch.mean(torch.tensor(val_losses)).item()

    def train(self):
        from tqdm import tqdm

        for epoch in range(self.config.max_epochs):
            # Training loop with progress bar
            loss = 0.0  # default in case train_loader is empty
            pbar = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=f"Epoch {epoch+1}/{self.config.max_epochs}",
            )

            for batch_idx, (inputs, targets) in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                loss = self.train_step(inputs, targets)

                # Update progress bar with current loss
                pbar.set_postfix({"loss": f"{loss:.4f}"})

                if batch_idx > 0 and batch_idx % self.config.eval_interval == 0:
                    val_loss = self.evaluate(self.val_loader)
                    pbar.set_postfix(
                        {"loss": f"{loss:.4f}", "val_loss": f"{val_loss:.4f}"}
                    )
                    print(
                        f"\n--- Epoch {epoch+1} | Step {batch_idx} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} ---"
                    )

                if batch_idx % self.config.gen_indx == 0 and batch_idx > 0:
                    prompt = (
                        self.config.start_context
                        if self.config.start_context
                        else "Garments are"
                    )
                    tokenizer = tiktoken.get_encoding("gpt2")
                    start_tokens = tokenizer.encode(
                        prompt, allowed_special={"<|endoftext|>"}
                    )
                    x = torch.tensor(
                        start_tokens, dtype=torch.long, device=self.device
                    )[None, :]
                    # generate output and decode it
                    result = self.generate(
                        x, max_new_tokens=50, context_size=self.config.context_length
                    )
                    print(tokenizer.decode(result[0].tolist()))

            self._save_checkpoint(epoch, loss)

    def generate(self, idx, max_new_tokens, context_size, eos_id=None):
        self.model.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad(), torch.amp.autocast(
                device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp
            ):
                logits = self.model(idx_cond)
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_id is not None and idx_next == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        self.model.train()
        return idx

    def generate_from_cache():
        pass


# if __name__ == "__main__":
#     from BLLMC.components.config import GPT_Config
#     from BLLMC.components.models import GPT2Model
#     from BLLMC.data.loader import create_dataloader
#     from BLLMC.components.trainer import LLMTrainer
#     import tiktoken

#     config = GPT_Config(compile=False)
#     model = GPT2Model(config)

#     with open(config.train_data_path, "r", encoding="utf-8") as f:

#     with open(config.val_data_path, "r", encoding="utf-8") as f:
#         val_data = f.read()

#     train_loader = create_dataloader(train_data, "gpt2", config)
#     val_loader = create_dataloader(val_data, "gpt2", config)

#     trainer = LLMTrainer(model, train_loader, val_loader, config)
#     trainer.train()

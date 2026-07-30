"""
Unit and integration tests for the BLLMC trainer, scheduler, and training pipeline.

Tests cover:
    - CosineWarmupScheduler learning rate behavior (warmup and cosine decay phases)
    - LLMTrainer initialization
    - Single train step execution
    - Evaluation loop
    - Checkpoint saving and loading
"""

import os
import shutil
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from BLLMC.components.config import gpt2_config
from BLLMC.components.base import ModelFactory, CosineWarmupScheduler
from BLLMC.components.trainer import LLMTrainer
from BLLMC.data.loader import create_dataloader


@pytest.fixture
def trainer_test_config(tmp_path):
    """Create a minimal GPT-2 config for fast testing, storing checkpoints in tmp_path."""
    ckpt_dir = os.path.join(tmp_path, "model_ckpt")
    return gpt2_config(
        emb_dim=32,
        n_heads=2,
        n_kv_heads=2,
        n_layers=1,
        context_length=16,
        max_length=16,
        stride=16,
        vocab_size=50257,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_epochs=1,
        warmup_steps=2,
        learning_rate=1e-3,
        min_lr=1e-5,
        checkpoint_dir=ckpt_dir,
        tokenizer_backend="tiktoken",
        tokenizer_model="gpt2",
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )


@pytest.fixture
def dummy_data():
    """Simple text dataset for tokenization."""
    return "Hello world! This is a simple test sequence to train a small model quickly." * 10


@pytest.fixture
def dataloaders(dummy_data, trainer_test_config):
    """Train and validation dataloaders."""
    train_loader = create_dataloader(dummy_data, "gpt2", trainer_test_config)
    val_loader = create_dataloader(dummy_data, "gpt2", trainer_test_config)
    return train_loader, val_loader


# ─── Scheduler Tests ──────────────────────────────────────────────


def test_cosine_warmup_scheduler():
    """Verify CosineWarmupScheduler correctly applies linear warmup and cosine decay."""
    # Setup dummy parameter and optimizer
    param = nn.Parameter(torch.zeros(10))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    
    warmup_steps = 3
    max_steps = 10
    min_lr = 1e-5
    
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=warmup_steps, max_steps=max_steps, min_lr=min_lr
    )
    
    # 1. Warmup Phase (Steps 1 to 3)
    # Step 1
    lr_1 = scheduler.step()
    assert lr_1 == pytest.approx(1e-3 * (1 / 3))
    assert optimizer.param_groups[0]["lr"] == lr_1
    
    # Step 2
    lr_2 = scheduler.step()
    assert lr_2 == pytest.approx(1e-3 * (2 / 3))
    
    # Step 3
    lr_3 = scheduler.step()
    assert lr_3 == pytest.approx(1e-3 * (3 / 3))
    
    # 2. Cosine Decay Phase (Steps 4 to 10)
    lr_4 = scheduler.step()
    assert lr_4 < 1e-3
    
    # End of schedule: should reach min_lr
    for _ in range(6):
        scheduler.step()
    lr_final = scheduler.get_lr()
    assert lr_final == pytest.approx(min_lr)


# ─── Trainer Tests ────────────────────────────────────────────────


def test_llm_trainer_initialization(trainer_test_config, dataloaders):
    """Verify LLMTrainer configures optimizer, loss, scheduler, and AMP properly."""
    train_loader, val_loader = dataloaders
    model = ModelFactory.create_model(trainer_test_config)
    
    trainer = LLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_test_config,
        device="cpu",
    )
    
    assert trainer.device == "cpu"
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert isinstance(trainer.scheduler, CosineWarmupScheduler)
    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    assert not trainer.use_amp  # AMP not enabled on CPU by default


def test_llm_trainer_train_step(trainer_test_config, dataloaders):
    """Verify that train_step computes loss and propagates gradients correctly."""
    train_loader, val_loader = dataloaders
    model = ModelFactory.create_model(trainer_test_config)
    trainer = LLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_test_config,
        device="cpu",
    )
    
    # Get a batch
    inputs, targets = next(iter(train_loader))
    
    # Clear parameters grads
    trainer.optimizer.zero_grad()
    
    # Perform training step (forward + backward only, gradient accumulation aware)
    loss = trainer.train_step(inputs, targets)
    
    assert isinstance(loss, float)
    assert loss > 0.0
    
    # Check that gradients were computed on some model parameter
    first_param = next(model.parameters())
    assert first_param.grad is not None


def test_llm_trainer_evaluate(trainer_test_config, dataloaders):
    """Verify validation loss computation."""
    train_loader, val_loader = dataloaders
    model = ModelFactory.create_model(trainer_test_config)
    trainer = LLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_test_config,
        device="cpu",
    )
    
    val_loss = trainer.evaluate(val_loader)
    
    assert isinstance(val_loss, float)
    assert val_loss > 0.0


def test_llm_trainer_checkpointing(trainer_test_config, dataloaders):
    """Verify checkpoint saving, cleanup of old checkpoints, and weights_only loading."""
    train_loader, val_loader = dataloaders
    model = ModelFactory.create_model(trainer_test_config)
    trainer = LLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_test_config,
        device="cpu",
    )
    
    # Ensure clean slate
    shutil.rmtree(trainer_test_config.checkpoint_dir, ignore_errors=True)
    
    # Save checkpoint for epoch 0
    trainer._save_checkpoint(epoch=0, loss=2.5)
    ckpt_path_0 = os.path.join(trainer_test_config.checkpoint_dir, "ckpt_epoch_0.pt")
    assert os.path.exists(ckpt_path_0)
    
    # Save checkpoint for epoch 1 (should remove epoch 0 checkpoints to save space)
    trainer._save_checkpoint(epoch=1, loss=2.1)
    ckpt_path_1 = os.path.join(trainer_test_config.checkpoint_dir, "ckpt_epoch_1.pt")
    assert os.path.exists(ckpt_path_1)
    assert not os.path.exists(ckpt_path_0)  # Verify epoch 0 checkpoint cleanup
    
    # Load and restore using weights_only
    checkpoint = trainer._load_checkpoint(ckpt_path_1)
    assert checkpoint["epoch"] == 1
    assert checkpoint["loss"] == pytest.approx(2.1)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint

    # Clean up test output
    shutil.rmtree(trainer_test_config.checkpoint_dir, ignore_errors=True)

import os
import sys
import torch
from torch.utils.data import DataLoader

from BLLMC.components.config import gptoss_config
from BLLMC.components.models import ModelFactory
from BLLMC.components.trainer import LLMTrainer
from BLLMC.data.dataset import download_dataset
from BLLMC.data.ingestion import DataIngestion
from BLLMC.data.loader import BanglaDataset
from BLLMC.data.tokenizer import get_tokenizer
from BLLMC.utils.logger import logger
from BLLMC.utils.exception import CustomException


def run_training():
    try:
        # 1. Load the GPT-OSS Configuration scaled down to fit on a single RTX 3090
                 # 1. Load the GPT-OSS Configuration scaled down to fit on a single RTX 3090
        config = gptoss_config(
            compile=True,
            batch_size=20,
            num_workers=0,
            max_epochs=5,
            n_layers=12,
            num_experts=4,
            emb_dim=512,
            n_heads=8,
            n_kv_heads=2,
            moe_hidden_dim=512,
            head_dim=64,
            dataset_path="dataset/bangla_dataset_v2.txt",
            train_data_path="dataset/bangla_train_v2.txt",
            val_data_path="dataset/bangla_val_v2.txt",
            test_data_path="dataset/bangla_test_v2.txt",
            )


        logger.info("Initializing GPT-OSS training pipeline...")

        # 2. Handle Dataset: Download if not present
        if not os.path.exists(config.dataset_path):
            logger.info(f"Dataset path '{config.dataset_path}' not found. Downloading...")
            download_dataset()
            # If the downloaded dataset is placed at a different path, update config
            if os.path.exists("dataset/bangla_dataset.txt"):
                config.dataset_path = "dataset/bangla_dataset.txt"

        # 3. Handle Data Ingestion / Splitting
        if not os.path.exists(config.train_data_path) or not os.path.exists(config.val_data_path):
            logger.info("Ingesting and splitting dataset...")
            ingestion = DataIngestion(config)
            ingestion.initiate_data_ingestion()

        # 4. Load train and validation text
        logger.info(f"Loading training data from: {config.train_data_path}")
        with open(config.train_data_path, "r", encoding="utf-8") as f:
            train_text = f.read()

        logger.info(f"Loading validation data from: {config.val_data_path}")
        with open(config.val_data_path, "r", encoding="utf-8") as f:
            val_text = f.read()

        # 5. Load Tokenizer
        logger.info(f"Loading tokenizer: backend={config.tokenizer_backend}, model={config.tokenizer_model}")
        tokenizer = get_tokenizer(config)

        # 6. Setup Dataset and Dataloaders
        # Using BanglaDataset directly with the loaded tokenizer strategy
        train_dataset = BanglaDataset(train_text, tokenizer, config)
        val_dataset = BanglaDataset(val_text, tokenizer, config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=config.drop_last,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=config.drop_last,
        )

        # 7. Build Model using the Model Factory
        logger.info("Building GPT-OSS model from config...")
        model = ModelFactory.create_model(config)

        # 8. Setup Trainer and Start Training
        trainer = LLMTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )

        logger.info("Starting training loop...")
        trainer.train()
        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Failed in training pipeline: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_training()

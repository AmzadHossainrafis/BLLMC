"""
src.BLLMC.data.ingestion

license : MIT
author :  Amzad hossain rafi
email : [EMAIL_ADDRESS]

log history :
    2026-05-16 12:15 PM : create the data ingestion
    2026-05-16 12:16 PM : add the config argument


"""

import os
import sys
from BLLMC.utils.logger import logger
from BLLMC.utils.exception import CustomException
from BLLMC.components.config import GPT_Config


class DataIngestion:
    def __init__(self, config: GPT_Config):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logger.info("Starting data ingestion process")
            dataset_path = self.config.dataset_path

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            logger.info(f"Reading dataset from {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Split the data
            total_len = len(text)
            train_len = int(total_len * self.config.train_split)
            val_len = int(total_len * self.config.val_split)

            train_text = text[:train_len]
            val_text = text[train_len : train_len + val_len]
            test_text = text[train_len + val_len :]

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            if self.config.val_data_path:
                os.makedirs(os.path.dirname(self.config.val_data_path), exist_ok=True)
            if self.config.test_data_path:
                os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)

            logger.info(f"Saving train data to {self.config.train_data_path}")
            with open(self.config.train_data_path, "w", encoding="utf-8") as f:
                f.write(train_text)

            if self.config.val_data_path:
                logger.info(f"Saving validation data to {self.config.val_data_path}")
                with open(self.config.val_data_path, "w", encoding="utf-8") as f:
                    f.write(val_text)

            if self.config.test_data_path:
                logger.info(f"Saving test data to {self.config.test_data_path}")
                with open(self.config.test_data_path, "w", encoding="utf-8") as f:
                    f.write(test_text)

            logger.info("Data ingestion completed successfully")

            return (
                self.config.train_data_path,
                self.config.val_data_path,
                self.config.test_data_path,
            )

        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)

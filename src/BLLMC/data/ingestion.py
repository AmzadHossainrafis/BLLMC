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

            logger.info(f"Measuring character count of dataset at {dataset_path}")
            total_chars = 0
            # Read in 1MB chunks to count characters memory-efficiently
            with open(dataset_path, "r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    total_chars += len(chunk)

            # Split ratios
            train_chars = int(total_chars * self.config.train_split)
            val_chars = int(total_chars * self.config.val_split)
            test_chars = total_chars - train_chars - val_chars

            logger.info(
                f"Total characters: {total_chars}. Splits: Train={train_chars}, Val={val_chars}, Test={test_chars}"
            )

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            if self.config.val_data_path:
                os.makedirs(os.path.dirname(self.config.val_data_path), exist_ok=True)
            if self.config.test_data_path:
                os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)

            logger.info("Streaming and splitting the dataset file...")
            with open(dataset_path, "r", encoding="utf-8") as f_in:
                # Write Train Data
                if train_chars > 0:
                    logger.info(f"Saving train data to {self.config.train_data_path}")
                    with open(
                        self.config.train_data_path, "w", encoding="utf-8"
                    ) as f_out:
                        chars_written = 0
                        while chars_written < train_chars:
                            chunk_size = min(1024 * 1024, train_chars - chars_written)
                            chunk = f_in.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            chars_written += len(chunk)

                # Write Val Data
                if val_chars > 0 and self.config.val_data_path:
                    logger.info(
                        f"Saving validation data to {self.config.val_data_path}"
                    )
                    with open(
                        self.config.val_data_path, "w", encoding="utf-8"
                    ) as f_out:
                        chars_written = 0
                        while chars_written < val_chars:
                            chunk_size = min(1024 * 1024, val_chars - chars_written)
                            chunk = f_in.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            chars_written += len(chunk)

                # Write Test Data
                if test_chars > 0 and self.config.test_data_path:
                    logger.info(f"Saving test data to {self.config.test_data_path}")
                    with open(
                        self.config.test_data_path, "w", encoding="utf-8"
                    ) as f_out:
                        # Write the rest of the file
                        while True:
                            chunk = f_in.read(1024 * 1024)
                            if not chunk:
                                break
                            f_out.write(chunk)

            logger.info("Data ingestion completed successfully")

            return (
                self.config.train_data_path,
                self.config.val_data_path,
                self.config.test_data_path,
            )

        except Exception as e:
            logger.error(f"Error in data ingestion: {str(e)}")
            raise CustomException(e, sys)

import gdown
import os
import sys
from BLLMC.utils.logger import logger
from BLLMC.utils.exception import CustomException


def download_dataset():
    """
    download the bangla dataset from google drive

    """

    file_id = "15zaO4sVBixE2Ww7Cbz3tyX_4o3zKDD33"
    output_file = "dataset/bangla_dataset.txt"
    if os.path.exists(output_file):
        logger.info("dataset already exist")

    else:
        try:
            logger.info("downloading dataset ")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            gdown.download(id=file_id, output=output_file, quiet=False)
            logger.info("dataset download complite ")

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    download_dataset()
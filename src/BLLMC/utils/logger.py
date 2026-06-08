"""
licence : mit
author : amzad hossain rafi
email : amzadhossain880@gmail.com

change log :
    8-6-2026 : implement logger in a singaleton pattern
"""

import os
import sys
import logging
import datetime as dt
from threading import Lock


class SingletonLogger:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SingletonLogger, cls).__new__(cls)
                    cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        today = dt.datetime.today().strftime("%Y-%m-%d")
        logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

        log_dir = "logs/"
        os.makedirs(log_dir, exist_ok=True)
        log_filepath = os.path.join(log_dir, f"running_logs_{today}.log")

        logging.basicConfig(
            level=logging.INFO,
            format=logging_str,
            handlers=[
                logging.FileHandler(log_filepath),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self._logger = logging.getLogger("BLLMC")

    def __getattr__(self, name):
        return getattr(self._logger, name)


logger = SingletonLogger()

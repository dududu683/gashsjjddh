import logging
import os
from datetime import datetime


class TrainLogger:
    """Logger for training process (console + file output)"""

    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

        # Configure logger
        self.logger = logging.getLogger('train_logger')
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        # File handler
        file_handler = logging.FileHandler(log_file)

        # Format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)


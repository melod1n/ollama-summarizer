import os
from logging.handlers import RotatingFileHandler

import logging

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "summary.log")

os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(name: str = "summary_logger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger(__name__)
log = setup_logger()
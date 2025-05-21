import os
from logging.handlers import RotatingFileHandler

import logging
from app.core.config import LOG_PATH

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def setup_logger(name: str = "summary_logger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

    return logger


# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger(__name__)
log = setup_logger()

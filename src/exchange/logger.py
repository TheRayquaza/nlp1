import logging
import os
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler  # Colorful logs (install with `pip install rich`)

os.makedirs("logs", exist_ok=True)

def get_logger(name: str = "recipe_api", log_level=logging.INFO) -> logging.Logger:
    """Create and configure a logger with a dynamically named log file."""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent duplicate handlers from being added
    if logger.hasHandlers():
        return logger

    # Log file named after the logger
    log_filename = f"logs/{name}.log"

    # Log format
    log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

    # File Handler (Rotating)
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=5 * 1024 * 1024, backupCount=3  # 5MB per file, 3 backups
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.setLevel(log_level)

    # Console Handler (Colorful Output)
    console_handler = RichHandler(
        rich_tracebacks=True, show_time=False, show_level=True
    )
    console_handler.setLevel(log_level)

    # Add Handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize Logger
logger = get_logger()

"""
Centralized logger configuration for the local_operator package.

- Sets log level from the LOG_LEVEL environment variable (default: WARNING).
- Applies log level to the main logger, requests, and common third-party loggers.
- Provides a function to retrieve the configured logger.

Usage:
    from local_operator.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import os
from typing import Optional

_LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _get_log_level() -> int:
    """
    Get the log level from the LOG_LEVEL environment variable.
    Defaults to logging.WARNING if not set or invalid.
    """
    level_str = os.environ.get("LOG_LEVEL", "WARNING").upper()
    return _LOG_LEVELS.get(level_str, logging.WARNING)


def _configure_logging(level: int) -> None:
    """
    Configure the root logger and common third-party loggers.
    Ensures all previous handlers are removed and the format is enforced.
    """
    # Remove all handlers from the root logger to prevent duplicate/conflicting configs
    root_logger = logging.getLogger()
    while root_logger.handlers:
        root_logger.handlers.pop()

    # Set up root logger with centralized format
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Ensure all handlers use the centralized format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for handler in root_logger.handlers:
        handler.setFormatter(formatter)

    # Set log level for requests, urllib3, langchain, and langchain.callbacks
    for lib_logger in ("requests", "urllib3", "langchain", "langchain.callbacks"):
        logging.getLogger(lib_logger).setLevel(level)


# Configure logging at import
_configure_logging(_get_log_level())


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retrieve a logger with the given name, configured with the centralized settings.

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)

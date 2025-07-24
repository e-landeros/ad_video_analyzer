"""
Centralized logging configuration for the Video Analyzer.

This module provides a consistent logging setup across the application,
with support for different log levels, formatting, and output destinations.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# More detailed format for debug mode
DEBUG_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

# Log levels mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_log_level(level_name: str) -> int:
    """
    Convert a log level name to its corresponding logging level.

    Args:
        level_name: The name of the log level (debug, info, warning, error, critical)

    Returns:
        int: The corresponding logging level

    Raises:
        ValueError: If the level name is not recognized
    """
    level_name = level_name.lower()
    if level_name not in LOG_LEVELS:
        valid_levels = ", ".join(LOG_LEVELS.keys())
        raise ValueError(
            f"Invalid log level: {level_name}. Valid levels are: {valid_levels}"
        )
    return LOG_LEVELS[level_name]


def configure_logging(
    level: Union[str, int] = "info",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 3,
    propagate: bool = True,
) -> None:
    """
    Configure the logging system for the Video Analyzer.

    Args:
        level: The log level (debug, info, warning, error, critical) or a logging level constant
        log_file: Optional path to a log file. If None, logs will only go to console
        log_format: Optional custom log format. If None, a default format will be used
        max_file_size: Maximum size of each log file in bytes (default: 10 MB)
        backup_count: Number of backup log files to keep (default: 3)
        propagate: Whether to propagate logs to parent loggers (default: True)

    Returns:
        None
    """
    # Convert string level to logging level if needed
    if isinstance(level, str):
        level = get_log_level(level)

    # Determine the log format based on the level
    if log_format is None:
        log_format = DEBUG_LOG_FORMAT if level <= logging.DEBUG else DEFAULT_LOG_FORMAT

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # Add file handler if a log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_file_size, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    # Configure the video_analyzer logger
    logger = logging.getLogger("video_analyzer")
    logger.setLevel(level)
    logger.propagate = propagate

    # Log the configuration
    logger.debug(
        f"Logging configured with level={logging.getLevelName(level)}, "
        f"log_file={log_file if log_file else 'None'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This is a convenience function that ensures all loggers are properly
    configured with the same settings.

    Args:
        name: The name of the logger, typically __name__

    Returns:
        logging.Logger: A configured logger instance
    """
    return logging.getLogger(name)

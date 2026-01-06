"""Logging configuration for StatGPT CLI."""

import logging
import sys
from datetime import date
from pathlib import Path

from statgpt.cli.settings import cli_settings


def setup_logging() -> logging.Logger:
    """Set up logging for the CLI.

    Configures logging to write to:
    - File: {cli_data_dir}/logs/cli-YYYY-MM-DD.log (always at DEBUG level)
    - Console: stderr (at configured LOG_LEVEL, only for WARNING+)

    Returns:
        The root CLI logger
    """

    # Create logger
    logger = logging.getLogger("statgpt.cli")
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter
    logger.propagate = False  # Don't propagate to root logger (common/config/logging.py)

    # Clear any existing handlers
    logger.handlers.clear()

    # Create logs directory
    log_dir = Path(cli_settings.cli_data_dir) / "logs"
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_file = log_dir / f"cli-{date.today().isoformat()}.log"

    # File handler - always DEBUG level for diagnostics
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(levelname)-8s | %(asctime)s | %(process)d | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - only WARNING and above (errors)
    # Regular output goes through Rich console, not logging
    console_handler = logging.StreamHandler(sys.stderr)
    console_level = getattr(logging, cli_settings.log_level, logging.INFO)
    # Only show WARNING+ on console to avoid noise
    console_handler.setLevel(max(console_level, logging.WARNING))
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.debug(f"Logging initialized. Log file: {log_file}")
    logger.debug(f"Log level: {cli_settings.log_level}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a CLI module.

    Args:
        name: Module name (will be prefixed with 'statgpt.cli.')

    Returns:
        Logger instance
    """
    if name.startswith("statgpt.cli."):
        return logging.getLogger(name)
    return logging.getLogger(f"statgpt.cli.{name}")

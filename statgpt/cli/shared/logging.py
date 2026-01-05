"""Logging configuration for StatGPT CLI."""

import logging
import sys
from pathlib import Path


def setup_logging() -> logging.Logger:
    """Set up logging for the CLI.

    Configures logging to write to:
    - File: {cli_data_dir}/cli.log (always at DEBUG level for diagnostics)
    - Console: stderr (at configured LOG_LEVEL, only for WARNING+)

    Returns:
        The root CLI logger
    """
    # Lazy import to avoid circular dependency with settings module
    from statgpt.cli.shared.settings import cli_settings

    # Create logger
    logger = logging.getLogger("statgpt.cli")
    logger.setLevel(logging.DEBUG)  # Capture all levels, handlers filter

    # Clear any existing handlers
    logger.handlers.clear()

    # Create log directory
    log_dir = Path(cli_settings.cli_data_dir)
    log_dir.mkdir(mode=0o700, exist_ok=True)
    log_file = log_dir / "cli.log"

    # File handler - always DEBUG level for diagnostics
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s - %(message)s",
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

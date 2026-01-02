"""StatGPT CLI - Interactive command-line interface for StatGPT administration."""

import asyncio
import sys

from statgpt.cli.commands import create_registry
from statgpt.cli.repl import run_repl
from statgpt.cli.shared.logging import setup_logging

__version__ = "1.0.0"


def main() -> None:
    """Main entry point for the StatGPT CLI."""
    # Initialize logging first
    logger = setup_logging()
    logger.info("StatGPT CLI starting")

    try:
        registry = create_registry()
        asyncio.run(run_repl(registry))
    except KeyboardInterrupt:
        logger.info("CLI interrupted by user")
        print("\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        logger.exception("CLI crashed with error")
        print(f"Error: {e}", file=sys.stderr)
        if "--debug" in sys.argv:
            raise
        sys.exit(1)


__all__ = ["main", "__version__"]

import logging
import logging.config
import sys

import uvicorn.logging
from aidial_sdk import logger as aidial_logger

from common.settings.logging import LoggingSettings


class SingleLineFormatter(uvicorn.logging.DefaultFormatter):
    def format(self, record):
        res = super().format(record).replace("\n", r"\n")
        return res


class LoggingConfig:

    LOGGING_SETTINGS = LoggingSettings()

    @classmethod
    def configure_logging(cls):
        # Making the uvicorn and dial_sdk loggers delegate its logging to the root logger
        for logger in [logging.getLogger("uvicorn"), aidial_logger]:
            logger.handlers = []
            logger.propagate = True

        # Setting up log levels
        for name in ["statgpt", "statgpt-ml", "admin_portal", "common", "__main__"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level)

        for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level_uvicorn)

        for name in ["httpcore", "httpx"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level_httpcore)

        for name in ["openai"]:
            logging.getLogger(name).setLevel(cls.LOGGING_SETTINGS.level_openai)

        # Configuring the root logger
        root = logging.getLogger()

        root_has_stderr_handler = any(
            isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr
            for handler in root.handlers
        )

        if not root_has_stderr_handler:
            formatter = uvicorn.logging.DefaultFormatter(
                fmt=cls.LOGGING_SETTINGS.format,
                datefmt=cls.LOGGING_SETTINGS.date_format,
                use_colors=True,
            )

            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(formatter)
            root.addHandler(handler)


LoggingConfig.configure_logging()
logger = logging.getLogger("statgpt")
multiline_logger = logging.getLogger("statgpt-ml")

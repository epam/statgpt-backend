import logging
import logging.config
import os

import uvicorn.logging

from common.utils.misc import str2bool


class SingleLineFormatter(uvicorn.logging.DefaultFormatter):
    def format(self, record):
        res = super().format(record).replace("\n", r"\n")
        return res


class LoggingConfig:
    DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] |%(process)d| %(pathname)s: %(message)s"
    DEFAULT_LOG_LEVEL = "INFO"

    CUSTOM_LOG_FORMAT = os.getenv("CUSTOM_LOG_FORMAT", DEFAULT_LOG_FORMAT)
    print(f"{CUSTOM_LOG_FORMAT=}")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    print(f"{LOG_LEVEL=}")

    LOG_MULTILINE_MODE_ENABLED = str2bool(os.getenv("LOG_MULTILINE_LOG_ENABLED", "false"))
    print(f"{LOG_MULTILINE_MODE_ENABLED=}")

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "logging.Formatter",
                "fmt": CUSTOM_LOG_FORMAT,
            },
            "single_line": {
                # single_line formatter will behave like default if LOG_MULTILINE_LOG_ENABLED
                "()": (
                    SingleLineFormatter if not LOG_MULTILINE_MODE_ENABLED else "logging.Formatter"
                ),
                "fmt": CUSTOM_LOG_FORMAT,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
            "console_single_line": {
                "class": "logging.StreamHandler",
                "formatter": "single_line",
            },
        },
        "root": {
            "handlers": ["console"],
            "level": LOG_LEVEL,
        },
        "loggers": {
            "statgpt": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
            "statgpt-ml": {  # statgpt multiline logger
                "handlers": ["console_single_line"],
                "level": LOG_LEVEL,
                "propagate": False,
            },
            # override third-party libs log format
            "uvicorn": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
            "uvicorn.access": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
            "uvicorn.error": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
            "openai": {"handlers": ["console"], "level": LOG_LEVEL, "propagate": False},
        },
    }

    @classmethod
    def configure_logging(cls):
        logging.config.dictConfig(cls.LOGGING_CONFIG)


LoggingConfig.configure_logging()
logger = logging.getLogger("statgpt")
multiline_logger = logging.getLogger("statgpt-ml")

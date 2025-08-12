import logging
import os
from enum import Enum

from common import utils
from common.config.logging import LoggingConfig


class DialAuthMode(str, Enum):
    USER_TOKEN = "user_token"
    """User token is passed to the DIAL API calls"""
    API_KEY = "api_key"
    """Configured API key is passed to the DIAL API calls"""


class DialAppConfig:
    APP_NAME: str = os.getenv("DIAL_APP_NAME", "talk-to-your-data")
    AUTH_MODE: DialAuthMode = DialAuthMode(
        os.getenv("DIAL_AUTH_MODE", DialAuthMode.USER_TOKEN).strip().lower()
    )
    LOG_LEVEL: str = os.getenv("DIAL_LOG_LEVEL", "INFO")

    SHOW_STAGE_SECONDS = utils.str2bool(os.getenv("DIAL_SHOW_STAGE_SECONDS", "false"))
    DIAL_SHOW_DEBUG_STAGES = utils.str2bool(os.getenv("DIAL_SHOW_DEBUG_STAGES", "false"))
    ENABLE_DEV_COMMANDS = utils.str2bool(os.getenv("ENABLE_DEV_COMMANDS", "false"))
    ENABLE_DIRECT_TOOL_CALLS = utils.str2bool(os.getenv("ENABLE_DIRECT_TOOL_CALLS", "false"))

    OFFICIAL_DATASET_LABEL = os.getenv("OFFICIAL_DATASET_LABEL", '⭐')

    # NOTE: below are not actually DIAL specific settings.
    # probably should be moved to a more general config file.
    SKIP_OUT_OF_SCOPE_CHECK = utils.str2bool(os.getenv("SKIP_OUT_OF_SCOPE_CHECK", "false"))
    # we can set default values for interceptable commands using env vars
    CMD_OUT_OF_SCOPE_ONLY = utils.str2bool(os.getenv("CMD_OUT_OF_SCOPE_ONLY", "false"))
    CMD_RAG_PREFILTER_ONLY = utils.str2bool(os.getenv("CMD_RAG_PREFILTER_ONLY", "false"))

    @classmethod
    def override_aidial_sdk_logger(cls, log_format: str):
        from aidial_sdk import logger as aidial_sdk_logger

        aidial_sdk_logger.propagate = False
        aidial_sdk_logger.setLevel(cls.LOG_LEVEL)

        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))

        aidial_sdk_logger.handlers = [handler]


DialAppConfig.override_aidial_sdk_logger(LoggingConfig.CUSTOM_LOG_FORMAT)

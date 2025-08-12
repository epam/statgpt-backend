import os

from langchain import globals as lc_globals

from common.config.llm_models import LLMModelsConfig
from common.utils.misc import str2bool

from .utils import get_int_env


class LangChainConfig:

    # Default chat completion settings
    DEFAULT_MODEL = os.getenv(
        "LANGCHAIN_DEFAULT_MODEL", default=LLMModelsConfig.GPT_4_TURBO_2024_04_09
    )
    DEFAULT_TEMPERATURE = float(os.getenv("LANGCHAIN_DEFAULT_TEMPERATURE", default="0"))
    DEFAULT_API_VERSION = os.getenv("LANGCHAIN_DEFAULT_API_VERSION", "2024-08-01-preview")
    DEFAULT_SEED = get_int_env("LANGCHAIN_DEFAULT_SEED", default=None)

    # Debugging settings
    VERBOSE = str2bool(os.getenv("LANGCHAIN_VERBOSE", "false"))
    DEBUG = str2bool(os.getenv("LANGCHAIN_DEBUG", "false"))
    USE_CUSTOM_LOGGER_CALLBACK = str2bool(
        os.getenv("LANGCHAIN_USE_CUSTOM_LOGGER_CALLBACK", "false")
    )

    @classmethod
    def configure(cls):
        lc_globals.set_verbose(LangChainConfig.VERBOSE)
        lc_globals.set_debug(LangChainConfig.DEBUG)


LangChainConfig.configure()

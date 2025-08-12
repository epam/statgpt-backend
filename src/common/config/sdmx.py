import os

from .utils import get_int_env


class SdmxConfig:
    CACHE_DIR = os.getenv("SDMX_CACHE_DIR")
    INDICATOR_COMBINATIONS_DIR = 'available_indicator_combinations'

    PORTAL_URL = os.getenv("SDMX_PORTAL_URL")

    MAX_RETRIES = get_int_env("SDMX_CLIENT_RETRY_COUNT", 5)
    RETRY_DELAY = get_int_env("SDMX_CLIENT_RETRY_DELAY", 3)

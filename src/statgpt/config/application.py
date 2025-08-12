import os
from enum import Enum


class AppMode(str, Enum):
    LOCAL = "LOCAL"
    DIAL = "DIAL"


class AppConfig:
    MODE = AppMode(os.getenv("APP_MODE", "DIAL"))

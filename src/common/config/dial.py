import os

from pydantic import SecretStr


class DialConfig:
    _URL: str = os.getenv("DIAL_URL", "http://localhost:8080")
    _API_KEY: SecretStr = SecretStr(os.getenv("DIAL_API_KEY", ""))

    @classmethod
    def get_url(cls) -> str:
        """
        Get the URL of the DIAL Core API, where this app is deployed.
        :return: URL of DIAL Core API
        """
        return cls._URL

    @classmethod
    def get_api_key(cls) -> SecretStr:
        """
        Get the API key for the DIAL Core API, where this app is deployed.
        :return: API key for DIAL Core API
        """
        return cls._API_KEY

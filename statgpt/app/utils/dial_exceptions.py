"""Custom Dial exceptions for the StatGPT application responses"""

import re

import openai
from aidial_sdk.exceptions import HTTPException as DialException

_DEPLOYMENT_PATH_PATTERN = re.compile(r"/openai/deployments/([^/]+)/")
_LIMIT_PATTERN = re.compile(r"(\w+) limit: (\d+) / (\d+) tokens")
_PERIOD_LABELS: dict[str, str] = {
    "Minute": "minute",
    "Day": "daily",
    "Week": "weekly",
    "Month": "monthly",
}


class RateLimitException(DialException):
    TYPE = "rate_limit_exceeded"

    def __init__(self, message: str, **kwargs) -> None:
        # NOTE: status_code is set to 500 instead of 429 because the DIAL Core does not forward 429 responses
        super().__init__(
            status_code=500,
            message=message,
            code="500",
            type=self.TYPE,
            **kwargs,
        )

    @classmethod
    def from_openai_error(cls, e: openai.RateLimitError) -> "RateLimitException":
        model = cls._get_underlying_model(e)
        retry_after = e.response.headers.get("retry-after")
        display_message = cls._get_display_message(e, retry_after)

        return cls(
            message=e.message,
            display_message=display_message,
            original_error=e.body,
            underlying_model=model,
            retry_after=retry_after,
        )

    @staticmethod
    def _get_underlying_model(e: openai.RateLimitError) -> str | None:
        try:
            url = str(e.response.request.url)
            match = _DEPLOYMENT_PATH_PATTERN.search(url)
            return match.group(1) if match else None
        except Exception:
            return None

    @classmethod
    def _get_display_message(cls, e: openai.RateLimitError, retry_after: str | None) -> str:
        try:
            retry_hint = f"in {cls._format_duration(int(retry_after))}" if retry_after else "later"
        except (ValueError, TypeError):
            retry_hint = "later"

        try:
            for match in _LIMIT_PATTERN.finditer(e.message):
                period, used, allowed = match.group(1), int(match.group(2)), int(match.group(3))
                if used > allowed:
                    label = _PERIOD_LABELS.get(period, period.lower())
                    return (
                        f"You've exceeded your {label} token limit for the underlying LLM model."
                        f" Please try again {retry_hint}"
                    )
        except Exception:
            pass
        return f"You've exceeded your token limit for the underlying LLM model. Please try again {retry_hint}"

    @staticmethod
    def _format_duration(seconds: int) -> str:
        if seconds >= 86400:
            return f"{seconds // 86400} days"
        if seconds >= 3600:
            return f"{seconds // 3600} hours"
        if seconds >= 60:
            return f"{seconds // 60} minutes"
        return f"{seconds} seconds"

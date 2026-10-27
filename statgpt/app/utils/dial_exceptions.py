"""Custom Dial exceptions for the StatGPT application responses"""

import re

import openai
from aidial_sdk.exceptions import HTTPException as DIALException

_DEPLOYMENT_PATH_PATTERN = re.compile(r"/openai/deployments/([^/]+)/")
_LIMIT_PATTERN = re.compile(r"(\w+) limit: (\d+) / (\d+) tokens")
_PERIOD_LABELS: dict[str, str] = {
    "Minute": "minute",
    "Day": "daily",
    "Week": "weekly",
    "Month": "monthly",
}
_DURATION_UNITS = [
    (86400, "day"),
    (3600, "hour"),
    (60, "minute"),
    (1, "second"),
]


def _get_underlying_deployment(e: openai.APIStatusError) -> str | None:
    try:
        url = str(e.response.request.url)
        match = _DEPLOYMENT_PATH_PATTERN.search(url)
        return match.group(1) if match else None
    except Exception:
        return None


class ForbiddenDeploymentException(DIALException):
    TYPE = "forbidden_deployment"

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(
            status_code=403,
            message=message,
            code="403",
            type=self.TYPE,
            **kwargs,
        )

    @classmethod
    def from_openai_error(cls, e: openai.PermissionDeniedError) -> "ForbiddenDeploymentException":
        deployment = _get_underlying_deployment(e)
        deployment_hint = f" '{deployment}'" if deployment else ""
        return cls(
            message=e.message,
            display_message=(
                f"You don't have access to the underlying deployment{deployment_hint}."
                " Please contact your administrator."
            ),
            original_error=e.body,
            underlying_deployment=deployment,
        )


class RateLimitException(DIALException):
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
        deployment = _get_underlying_deployment(e)
        retry_after = e.response.headers.get("retry-after")
        exceeded_limits = cls._get_exceeded_limits(e)
        display_message = cls._get_display_message(exceeded_limits, retry_after)

        return cls(
            message=e.message,
            display_message=display_message,
            original_error=e.body,
            underlying_deployment=deployment,
            retry_after=retry_after,
            exceeded_limit=exceeded_limits,
        )

    @staticmethod
    def _get_exceeded_limits(e: openai.RateLimitError) -> list[str]:
        try:
            return [
                _PERIOD_LABELS.get(match.group(1), match.group(1).lower())
                for match in _LIMIT_PATTERN.finditer(e.message)
                if int(match.group(2)) > int(match.group(3))
            ]
        except Exception:
            return []

    @classmethod
    def _get_display_message(cls, exceeded_limits: list[str], retry_after: str | None) -> str:
        try:
            retry_hint = f"in {cls._format_duration(int(retry_after))}" if retry_after else "later"
        except (ValueError, TypeError):
            retry_hint = "later"

        if exceeded_limits:
            labels = ", ".join(exceeded_limits)
            return (
                f"You've exceeded your {labels} token limit for the underlying deployment."
                f" Please try again {retry_hint}"
            )
        return f"You've exceeded your token limit for the underlying deployment. Please try again {retry_hint}"

    @staticmethod
    def _format_duration(total_seconds: int) -> str:
        parts: list[str] = []
        remaining = total_seconds
        for unit_seconds, unit_name in _DURATION_UNITS:
            if remaining >= unit_seconds:
                value = remaining // unit_seconds
                remaining %= unit_seconds
                suffix = "s" if value != 1 else ""
                parts.append(f"{value} {unit_name}{suffix}")
                if len(parts) == 2:
                    break
        return " ".join(parts) if parts else f"{total_seconds} seconds"

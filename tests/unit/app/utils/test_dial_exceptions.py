"""Unit tests for the DIAL exceptions StatGPT responds with."""

import httpx
import openai

from statgpt.app.utils.dial_exceptions import RateLimitException

_MESSAGE = "Hit cost rate limit. Minute limit: $0.00 / $1.00. Day limit: $0.05 / $0.01."


def _rate_limit_error(headers: dict[str, str]) -> openai.RateLimitError:
    request = httpx.Request("POST", "http://dial.test/openai/deployments/gpt-4o/chat/completions")
    response = httpx.Response(429, headers=headers, request=request)
    return openai.RateLimitError(_MESSAGE, response=response, body={"message": _MESSAGE})


class TestRateLimitException:
    def test_keeps_500_status_with_429_code(self) -> None:
        exc = RateLimitException.from_openai_error(_rate_limit_error({"retry-after": "53932"}))

        assert exc.status_code == 500
        assert exc.code == "429"
        assert exc.type == "rate_limit_exceeded"
        assert exc.json_error()["error"]["code"] == "429"

    def test_tells_openai_sdk_callers_not_to_retry(self) -> None:
        exc = RateLimitException.from_openai_error(_rate_limit_error({"retry-after": "53932"}))

        assert exc.headers == {"x-should-retry": "false", "Retry-After": "53932"}

    def test_omits_retry_after_when_unknown(self) -> None:
        exc = RateLimitException.from_openai_error(_rate_limit_error({}))

        assert exc.headers == {"x-should-retry": "false"}


class TestRateLimitDisplayMessage:
    @staticmethod
    def _error(message: str, headers: dict[str, str]) -> openai.RateLimitError:
        request = httpx.Request(
            "POST", "http://dial.test/openai/deployments/gpt-4o/chat/completions"
        )
        response = httpx.Response(429, headers=headers, request=request)
        return openai.RateLimitError(message, response=response, body={"message": message})

    def test_names_the_daily_limit_and_when_it_resets(self) -> None:
        error = self._error("Day limit: 120000 / 100000 tokens", {"retry-after": "3700"})

        exc = RateLimitException.from_openai_error(error)

        assert exc.display_message == (
            "You've exceeded your daily token limit for the underlying deployment."
            " The limit resets in 1 hour 1 minute."
        )

    def test_asks_to_wait_without_retry_after(self) -> None:
        error = self._error("Day limit: 120000 / 100000 tokens", {})

        exc = RateLimitException.from_openai_error(error)

        assert exc.display_message == (
            "You've exceeded your daily token limit for the underlying deployment."
            " Please wait for the limit to reset."
        )

    def test_without_known_limits(self) -> None:
        exc = RateLimitException.from_openai_error(_rate_limit_error({}))

        assert exc.display_message == (
            "You've exceeded your token limit for the underlying deployment."
            " Please wait for the limit to reset."
        )

    def test_never_says_try_again(self) -> None:
        exc = RateLimitException.from_openai_error(_rate_limit_error({"retry-after": "53932"}))

        assert "try again" not in exc.display_message.lower()

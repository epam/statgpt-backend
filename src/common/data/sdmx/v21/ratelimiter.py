import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager

from common.data.sdmx.common.config import SdmxRateLimitsConfig

_log = logging.getLogger(__name__)


class RateLimiter(ABC):
    @abstractmethod
    @asynccontextmanager
    async def limiter(self):
        pass


class NoOpRateLimiter(RateLimiter):
    @asynccontextmanager
    async def limiter(self):
        yield


class FixedConcurrencyRateLimiter(RateLimiter):
    def __init__(self, limit: int):
        self._semaphore = asyncio.Semaphore(limit)
        self._limit = limit
        _log.debug(f"Created FixedConcurrencyRateLimiter with limit={limit}")

    @asynccontextmanager
    async def limiter(self):
        current_value = self._semaphore._value
        _log.debug(f"Acquiring semaphore (limit={self._limit}, available={current_value})")
        async with self._semaphore:
            _log.debug(
                f"Semaphore acquired (limit={self._limit}, available={self._semaphore._value})"
            )
            try:
                yield
            finally:
                _log.debug(
                    f"Releasing semaphore (limit={self._limit}, will be available={self._semaphore._value + 1})"
                )


class BaseSdmxRateLimiter(ABC):

    @abstractmethod
    @asynccontextmanager
    async def structure_limiter(self):
        pass

    @abstractmethod
    @asynccontextmanager
    async def availability_limiter(self):
        pass

    @abstractmethod
    @asynccontextmanager
    async def data_limiter(self):
        pass


def create_limiter(limit: int | None) -> RateLimiter:
    if limit is None or limit <= 0:
        _log.debug(f"Creating NoOpRateLimiter (limit={limit})")
        return NoOpRateLimiter()
    _log.debug(f"Creating FixedConcurrencyRateLimiter with limit={limit}")
    return FixedConcurrencyRateLimiter(limit)


class SdmxRateLimiter(BaseSdmxRateLimiter):
    def __init__(
        self,
        structure_limiter: RateLimiter,
        availability_limiter: RateLimiter,
        data_limiter: RateLimiter,
    ):
        self._structure_limiter = structure_limiter
        self._availability_limiter = availability_limiter
        self._data_limiter = data_limiter
        _log.debug("Created SdmxRateLimiter with structure, availability, and data limiters")

    @asynccontextmanager
    async def structure_limiter(self):
        async with self._structure_limiter.limiter():
            yield

    @asynccontextmanager
    async def availability_limiter(self):
        async with self._availability_limiter.limiter():
            yield

    @asynccontextmanager
    async def data_limiter(self):
        async with self._data_limiter.limiter():
            yield

    @classmethod
    def from_config(cls, rate_limits: SdmxRateLimitsConfig) -> "SdmxRateLimiter":
        _log.debug(f"Creating SdmxRateLimiter from config: {rate_limits}")
        availability_and_data_rate_limiter = create_limiter(
            rate_limits.get_availability_and_data_requests_concurrency()
        )
        return cls(
            structure_limiter=create_limiter(rate_limits.get_structure_requests_concurrency()),
            availability_limiter=availability_and_data_rate_limiter,
            data_limiter=availability_and_data_rate_limiter,
        )


class SdmxRateLimiterFactory:

    _RATE_LIMITERS: dict[str, tuple[SdmxRateLimiter, SdmxRateLimitsConfig]] = {}
    _lock = asyncio.Lock()

    @classmethod
    def _should_recreate_limiter(
        cls, source_id: str, existing_config: SdmxRateLimitsConfig, new_config: SdmxRateLimitsConfig
    ) -> bool:
        """Check if rate limiter should be recreated due to config changes."""
        if existing_config != new_config:
            _log.info(
                f"Rate limits changed for {source_id=}. "
                f"Old config: {existing_config}, New config: {new_config}. "
            )
            return True
        _log.debug(f"Rate limits unchanged for {source_id=}")
        return False

    @classmethod
    async def get(cls, source_id: str, rate_limits: SdmxRateLimitsConfig) -> SdmxRateLimiter:
        _log.debug(f"Requesting rate limiter for {source_id=}")
        async with cls._lock:
            if source_id not in cls._RATE_LIMITERS:
                _log.info(f"Creating new rate limiter for {source_id=} with {rate_limits=}")
                rate_limiter = SdmxRateLimiter.from_config(rate_limits)
                cls._RATE_LIMITERS[source_id] = (rate_limiter, rate_limits)
            else:
                existing_limiter, existing_config = cls._RATE_LIMITERS[source_id]
                if cls._should_recreate_limiter(source_id, existing_config, rate_limits):
                    _log.info(f"Creating new rate limiter for {source_id=}")
                    rate_limiter = SdmxRateLimiter.from_config(rate_limits)
                    cls._RATE_LIMITERS[source_id] = (rate_limiter, rate_limits)
                else:
                    _log.debug(f"Using existing rate limiter for {source_id=}")
                    rate_limiter = existing_limiter
            return rate_limiter

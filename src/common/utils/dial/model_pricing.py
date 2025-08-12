from datetime import datetime, timedelta
from typing import ClassVar

from pydantic import ValidationError

from common.auth.auth_context import AuthContext
from common.config import DialConfig
from common.config import multiline_logger as logger
from common.schemas.dial import Pricing
from common.utils import DialCore

_CACHE: dict[str, tuple[datetime, Pricing]] = {}


class ModelPricingAuthContext(AuthContext):

    @property
    def is_system(self) -> bool:
        return False

    @property
    def dial_access_token(self) -> str | None:
        return None

    @property
    def api_key(self) -> str:
        return DialConfig.get_api_key().get_secret_value()


class ModelPricingGetter:
    CACHE_EXPIRATION_WINDOW: ClassVar[timedelta] = timedelta(days=1)

    def __init__(self, dial_core: DialCore):
        self._dial_core = dial_core

    async def get_model_pricing(self, model: str) -> Pricing | None:
        if pricing := self._get_pricing_from_cache(model):
            return pricing

        if pricing := await self._load_pricing(model):
            _CACHE[model] = (datetime.now(), pricing)
            return pricing

        return None

    def _get_pricing_from_cache(self, model: str) -> Pricing | None:
        if model in _CACHE:
            last_updated, pricing = _CACHE[model]
            if datetime.now() - last_updated < self.CACHE_EXPIRATION_WINDOW:
                return pricing
        return None

    async def _load_pricing(self, model: str) -> Pricing | None:
        try:
            model_data = await self._dial_core.get_model_by(name=model)
        except Exception as e:
            logger.error(f"Failed to fetch model data for model {model}: {e}")
            return None

        if "pricing" not in model_data:
            return None

        try:
            return Pricing.model_validate(model_data["pricing"])
        except ValidationError as e:
            logger.info(f"{model_data=}")
            logger.error(f"Failed to validate pricing for model {model}: {e}")
            return None

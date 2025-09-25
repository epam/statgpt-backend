from pydantic import ValidationError

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.schemas.dial import Pricing
from common.settings.dial import dial_settings
from common.utils import Cache, DialCore

_CACHE: Cache[Pricing] = Cache(ttl=24 * 3600)  # 24 hours


class ModelPricingAuthContext(AuthContext):

    @property
    def is_system(self) -> bool:
        return False

    @property
    def dial_access_token(self) -> str | None:
        return None

    @property
    def api_key(self) -> str:
        return dial_settings.api_key.get_secret_value()


class ModelPricingGetter:

    def __init__(self, dial_core: DialCore):
        self._dial_core = dial_core

    async def get_model_pricing(self, model: str) -> Pricing | None:
        if pricing := _CACHE.get(model):
            return pricing

        if pricing := await self._load_pricing(model):
            _CACHE.set(model, pricing)
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

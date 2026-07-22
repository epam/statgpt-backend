from aidial_client import AsyncDial
from aidial_client.types.model import ModelPricing

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.settings.dial import dial_settings
from statgpt.common.utils import TtlCache

_CACHE: TtlCache[ModelPricing] = TtlCache(ttl=24 * 3600)  # 24 hours


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

    def __init__(self, dial: AsyncDial):
        self._dial = dial

    async def get_model_pricing(self, model: str) -> ModelPricing | None:
        if pricing := _CACHE.get(model):
            return pricing

        if pricing := await self._load_pricing(model):
            _CACHE.set(model, pricing)
            return pricing

        return None

    async def _load_pricing(self, model: str) -> ModelPricing | None:
        try:
            model_data = await self._dial.model.get(model)
        except Exception as e:
            logger.error(f"Failed to fetch model data for model {model}: {e}")
            return None

        return model_data.pricing

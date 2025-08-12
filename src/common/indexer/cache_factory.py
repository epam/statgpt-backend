from common.config.logging import logger
from common.indexer.cache import Cache


class CacheFactory:
    _cache = Cache()

    @classmethod
    def get_instance(cls) -> Cache:
        logger.info(f"[cache] factory: {id(cls)}, cache: {id(cls._cache)}")
        return cls._cache

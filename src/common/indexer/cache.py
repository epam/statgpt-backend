import time

from common.config.logging import logger


class Cache:
    DEFAULT_TTL_SEC = 15 * 60

    def __init__(self):
        self._items = {}
        self._ttl = {}

    def put(self, key, value, ttl_sec=DEFAULT_TTL_SEC):
        self._items[key] = value
        self._ttl[key] = time.time() + ttl_sec

    def get(self, search_key):
        now = time.time()
        logger.info("[cache] items:")
        for key in list(self._ttl.keys()):
            if now > self._ttl[key]:
                del self._ttl[key]
                del self._items[key]
                continue
            logger.info(f"- {key}")

        if search_key not in self._ttl or search_key not in self._items:
            return None, None
        return self._items[search_key], self._ttl[search_key] - now

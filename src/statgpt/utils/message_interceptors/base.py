import logging
import typing as t
from abc import ABC, abstractmethod

from common.schemas.dial import Message as DialMessage

_log = logging.getLogger(__name__)


class BaseMessageInterceptor(ABC):

    @abstractmethod
    async def process_messages(
        self, messages: list[DialMessage], state: dict[str, t.Any]
    ) -> list[DialMessage]:
        pass

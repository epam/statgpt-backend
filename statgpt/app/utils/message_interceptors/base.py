import logging
from abc import ABC, abstractmethod

from aidial_sdk.chat_completion import Message as DialMessage

from statgpt.app.schemas.state import State

_log = logging.getLogger(__name__)


class BaseMessageInterceptor(ABC):

    @abstractmethod
    async def process_messages(
        self, messages: list[DialMessage], state: State
    ) -> list[DialMessage]:
        pass

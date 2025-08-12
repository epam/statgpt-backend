from abc import ABC, abstractmethod

from langchain_core.runnables import Runnable

from common.schemas import ChannelConfig, DataQueryDetails
from statgpt.utils.request_context import RequestContext


class BaseDataQueryFactory(ABC):

    def __init__(self, config: DataQueryDetails, channel_config: ChannelConfig):
        self._config = config
        self._channel_config = channel_config

    @abstractmethod
    async def create_chain(self, request_context: RequestContext) -> Runnable:
        pass

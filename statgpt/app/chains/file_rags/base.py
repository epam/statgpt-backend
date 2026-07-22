from abc import ABC, abstractmethod

from langchain_core.runnables import Runnable

from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import FileRagTool as FileRagToolConfig


class BaseRAGFactory(ABC):
    FIELD_RESPONSE = 'response'
    FIELD_ANSWERED_BY = 'answered_by'
    FIELD_ARTIFACT = 'file_rag_artifact'

    def __init__(self, tool_config: FileRagToolConfig, channel_config: ChannelConfig):
        self._tool_config = tool_config
        self._channel_config = channel_config

    @abstractmethod
    async def create_chain(self) -> Runnable:
        pass

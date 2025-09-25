from abc import ABC, abstractmethod

from langchain_core.runnables import Runnable

from common.schemas import ChannelConfig
from common.schemas import FileRagTool as FileRagToolConfig


class BaseRAGFactory(ABC):
    COMMAND = "!general"

    FIELD_RESPONSE = 'response'
    FIELD_ANSWERED_BY = 'answered_by'
    FIELD_ARTIFACT = 'file_rag_artifact'

    def __init__(self, tool_config: FileRagToolConfig, channel_config: ChannelConfig):
        self._tool_config = tool_config
        self._channel_config = channel_config

    @classmethod
    def list_commands(cls) -> list[str]:
        return [cls.COMMAND]

    @abstractmethod
    async def create_chain(self) -> Runnable:
        pass

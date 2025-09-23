from abc import ABC, abstractmethod

from langchain_core.runnables import Runnable

from common.schemas.data_query_tool import SpecialDimensionsProcessor


class SpecialDimensionChainFactoryBase(ABC):
    def __init__(self, processor: SpecialDimensionsProcessor):
        self._processor = processor
        self._name = type(self).__name__

    @abstractmethod
    def create_chain(self) -> Runnable:
        raise NotImplementedError

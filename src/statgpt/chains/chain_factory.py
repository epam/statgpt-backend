from abc import ABC, abstractmethod

from langchain_core.runnables import Runnable


class ChainFactory(ABC):
    @abstractmethod
    def create_chain(self) -> Runnable:
        pass

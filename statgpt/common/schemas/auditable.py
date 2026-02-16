from abc import ABC, abstractmethod


class Auditable(ABC):
    @abstractmethod
    def get_entity_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_entity_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_state_after(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_item_id(self) -> int:
        raise NotImplementedError

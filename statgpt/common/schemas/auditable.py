from __future__ import annotations

from abc import ABC, abstractmethod


class Auditable(ABC):
    @abstractmethod
    def get_entity_id(self) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def get_entity_name(self) -> str | None:
        raise NotImplementedError

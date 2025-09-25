import re
import typing as t
from abc import ABC, abstractmethod

from .enums import EntityType


class BaseEntity(ABC):

    @property
    @abstractmethod
    def entity_type(self) -> EntityType:
        pass

    @property
    @abstractmethod
    def entity_id(self) -> str:
        pass

    @property
    @abstractmethod
    def source_id(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> t.Optional[str]:
        pass

    def __str__(self):
        return f"{self.entity_type.value}={self.entity_id}({self.name})"

    def __repr__(self):
        return str(self)

    def get_file_name(self) -> str:
        result = re.sub(r'\W+', '_', self.entity_id).lower()
        result = re.sub(r'(^_+)|(_+$)', '', result)
        return result

    # below we implement __eq__ and __hash__ methods
    # to allow using any BaseEntity instances as dict keys and set items.

    def __eq__(self, other):
        if not isinstance(other, BaseEntity):
            return NotImplemented
        # here, self and other could be different BaseEntity subclasses.
        # thus we use both entity_id and entity_type to compare them.
        res = self.entity_id == other.entity_id and self.entity_type == other.entity_type
        return res

    def __hash__(self):
        return hash((self.entity_id, self.entity_type))

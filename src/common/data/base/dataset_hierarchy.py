from abc import ABC, abstractmethod
from collections.abc import Iterable

from pydantic import BaseModel


class HierarchyItem(BaseModel):
    entity_id: str
    name: str
    parent_entity_id: str | None
    is_dataset: bool

    def __hash__(self):
        return hash(self.entity_id)

    def __eq__(self, other):
        if not isinstance(other, HierarchyItem):
            return NotImplemented

        return self.entity_id == other.entity_id


TREE_TYPE = dict[HierarchyItem | None, 'TREE_TYPE']


class DatasetHierarchy:

    def __init__(self, items: Iterable[HierarchyItem]):
        self._items = {x.entity_id: x for x in items}

    def __getitem__(self, entity_id: str) -> HierarchyItem:
        if entity_id not in self._items:
            raise KeyError(f"Hierarchy item with entity_id {entity_id!r} not found.")
        return self._items[entity_id]

    def datasets(self) -> list[HierarchyItem]:
        return [x for x in self._items.values() if x.is_dataset]

    def get_children(self, parent_entity_id: str) -> list[HierarchyItem]:
        return [x for x in self._items.values() if x.parent_entity_id == parent_entity_id]

    def to_tree(self) -> TREE_TYPE:
        def _build_tree(parent_id: str | None) -> TREE_TYPE:
            parent = self._items[parent_id] if parent_id else None
            tree: TREE_TYPE = {}
            for item in self._items.values():
                if item.parent_entity_id == parent_id:
                    tree[parent] = _build_tree(item.entity_id)
            return tree

        return _build_tree(None)


class DatasetHierarchyCreatorABC(ABC):

    def __init__(self, handler, auth_context):
        self._handler = handler
        self._auth_context = auth_context

    @abstractmethod
    async def create_hierarchy(self) -> DatasetHierarchy:
        pass


class DefaultDatasetHierarchyCreator(DatasetHierarchyCreatorABC):

    async def create_hierarchy(self) -> DatasetHierarchy:
        # TODO: implement default hierarchy creation logic
        raise NotImplementedError("Not implemented yet")

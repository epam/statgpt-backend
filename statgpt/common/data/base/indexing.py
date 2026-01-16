import json
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from statgpt.common.utils import crc32_hash


class IndexingField:
    """Marker class to indicate a field should be included in indexing hash computation.

    Use with Annotated to mark fields:
    ```
        name: Annotated[str, IndexingField()] = Field(...)
    ```

    Marked fields are recursively collected when computing the indexing hash.
    For nested `BaseModel` fields, only their marked fields are included.

    IMPORTANT: When a child class redefines a field, it must explicitly include
    the `IndexingField` marker - markers are NOT inherited through field redefinition:
    ```
        class Parent(BaseModel):
            field: Annotated[str, IndexingField()] = "value"

        class Child(Parent):
            # WRONG - marker is lost:
            field: str = "other"

            # CORRECT - marker preserved:
            field: Annotated[str, IndexingField()] = "other"
    ```
    """

    def __repr__(self) -> str:
        return "IndexingField()"


def _has_indexing_marker(field_info: FieldInfo) -> bool:
    """Check if a field has the IndexingField marker in its metadata."""
    return any(isinstance(meta, IndexingField) for meta in field_info.metadata)


def _collect_indexing_fields(model: BaseModel) -> Generator[tuple[str, Any], None, None]:
    """Recursively collect all fields marked for indexing from a model.

    Yields tuples of `(field_path, value)` for deterministic serialization.

    For nested `BaseModel` fields that are marked:
    - Recursively collect their marked fields (granular control)
    - If nested model has no marked fields, include full model dump

    For `dict[str, BaseModel]` fields (like dimensions):
    - Sort by key for determinism
    - Recursively collect marked fields from each value
    """
    for field_name, field_info in type(model).model_fields.items():
        if not _has_indexing_marker(field_info):
            continue

        value = getattr(model, field_name)

        if value is None:
            yield field_name, None
        elif isinstance(value, BaseModel):
            # Recursively collect marked fields from nested model
            nested_fields = list(_collect_indexing_fields(value))
            for nested_path, nested_value in nested_fields:
                yield f"{field_name}.{nested_path}", nested_value
        elif isinstance(value, dict):
            # Handle dict[str, BaseModel] like dimensions
            dict_items = {}
            for k, v in sorted(value.items()):  # Sort keys for determinism
                if isinstance(v, BaseModel):
                    nested = dict(_collect_indexing_fields(v))
                    dict_items[k] = nested if nested else v.model_dump(mode="json")
                else:
                    dict_items[k] = v
            yield field_name, dict_items
        elif isinstance(value, (list, tuple)):
            # Handle collections of models
            items = []
            for item in value:
                if isinstance(item, BaseModel):
                    nested = dict(_collect_indexing_fields(item))
                    items.append(nested if nested else item.model_dump(mode="json"))
                else:
                    items.append(item)
            yield field_name, items
        else:
            yield field_name, value


def compute_indexing_hash(model: BaseModel) -> str:
    """Compute a deterministic CRC32 hash from all IndexingField-marked fields.

    The hash is computed by:
    1. Collecting all marked fields recursively
    2. Serializing to JSON with sorted keys for determinism
    3. Computing CRC32 hash of the JSON string

    Args:
        model: A Pydantic BaseModel instance with IndexingField markers

    Returns:
        String representation of the CRC32 hash
    """
    fields = dict(_collect_indexing_fields(model))
    # sort_keys=True ensures deterministic ordering at all nesting levels
    json_str = json.dumps(fields, sort_keys=True)
    return str(crc32_hash(json_str))


class IndexingHashMixin:
    """Mixin that provides indexing_hash property for Pydantic models.

    Classes using this mixin should mark fields with IndexingField() in Annotated.
    The mixin recursively collects all marked fields and computes a unified hash.

    Example:
        class MyConfig(BaseModel, IndexingHashMixin):
            name: Annotated[str, IndexingField()] = Field(...)
            description: str = Field(...)  # Not included in hash

        config = MyConfig(name="test", description="ignored")
        print(config.indexing_hash)  # Computed from 'name' only
    """

    @property
    def indexing_hash(self) -> str:
        """Compute hash from all fields marked with IndexingField."""
        if not isinstance(self, BaseModel):
            raise TypeError("IndexingHashMixin must be used with Pydantic BaseModel")
        return compute_indexing_hash(self)

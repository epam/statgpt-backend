import json
from collections.abc import Generator
from typing import Any, Self, TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from statgpt.common.utils import crc32_hash

_T = TypeVar("_T", bound=BaseModel)


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
    """Mixin that provides indexing hash and config merging for Pydantic models.

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

    def with_non_indexing_fields_from(self, source: Self) -> Self:
        """Return a new instance with non-indexing fields taken from source.

        Creates a copy of this model where:
        - IndexingField-marked fields: preserved from self
        - Non-marked fields: taken from source
        - Nested structures: recursively merged

        Args:
            source: Model instance to take non-indexing field values from

        Returns:
            New model instance with merged fields
        """
        if not isinstance(self, BaseModel):
            raise TypeError("IndexingHashMixin must be used with Pydantic BaseModel")
        return _merge_models(resolved=self, current=source)  # type: ignore[return-value]


def _merge_models(*, resolved: _T, current: _T) -> _T:
    """Merge two model instances based on IndexingField markers.

    - IndexingField-marked fields: taken from resolved
    - Non-marked fields: taken from current
    - Nested BaseModel fields: recursively merged
    - dict[str, BaseModel] fields: merged per-key using resolved keys
    """
    model_class = type(resolved)
    updates: dict[str, Any] = {}

    for field_name, field_info in model_class.model_fields.items():
        current_val = getattr(current, field_name)
        resolved_val = getattr(resolved, field_name)

        if _has_indexing_marker(field_info):
            if resolved_val is None:
                updates[field_name] = None
            elif isinstance(resolved_val, BaseModel):
                if current_val is None:
                    updates[field_name] = resolved_val
                else:
                    updates[field_name] = _merge_models(resolved=resolved_val, current=current_val)
            elif isinstance(resolved_val, dict):
                # dict[str, BaseModel] like dimensions - merge per key
                merged_dict: dict[str, Any] = {}
                for k, v in resolved_val.items():
                    if isinstance(v, BaseModel):
                        current_item = current_val.get(k) if current_val else None
                        if current_item is not None:
                            merged_dict[k] = _merge_models(resolved=v, current=current_item)
                        else:
                            merged_dict[k] = v
                    else:
                        merged_dict[k] = v
                updates[field_name] = merged_dict
            else:
                updates[field_name] = resolved_val
        else:
            updates[field_name] = current_val

    return model_class.model_construct(**updates)

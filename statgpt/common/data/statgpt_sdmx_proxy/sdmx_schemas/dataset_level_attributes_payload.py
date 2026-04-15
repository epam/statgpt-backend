"""
Pydantic shapes for the *dataset-level attributes* slice of an SDMX-JSON data message.

StatGPT SDMX proxy SDMX-JSON vs a typical registry-style SDMX-JSON (same logical slice):

- **Key spelling under ``attributes``**: registry payloads often use camelCase (for example
  ``dataSet``); the proxy may use lowercase ``dataset`` for the parallel component list.
- **Component metadata**: the proxy tends to attach richer descriptive fields on each
  attribute component (human-readable names, long descriptions, varied relationship
  shapes). Registry components are often minimal (identifiers, mandatory flags, compact
  relationship stubs).
- **``dataSets[].attributes`` slot encoding**: both use an array aligned with the ordered
  dataset-level attribute components. The proxy frequently stores an integer index into
  each component's ``values`` list, where entries are objects with several optional text
  fields. A registry may inline literals (strings, lists of localized text objects,
  ``null`` placeholders) or omit trailing positions when the encoder compacts the array—
  so the same logical row can mix integers, in-band text, and missing positions.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class DatasetLevelDataSet(BaseModel):
    """Single SDMX-JSON dataset entry; only ``attributes`` is used for this parser."""

    model_config = ConfigDict(extra="ignore")

    attributes: list[Any] = Field(default_factory=list)


class DatasetLevelAttributesData(BaseModel):
    """The ``data`` object (or the payload root when ``data`` is absent)."""

    model_config = ConfigDict(extra="ignore")

    data_sets: list[DatasetLevelDataSet] = Field(alias="dataSets", default_factory=list)
    structures: list[Any] = Field(default_factory=list)


class DatasetAttributeComponentDef(BaseModel):
    """One dataset-level attribute component from ``structures[].attributes``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    values: list[Any] = Field(default_factory=list)


def _dataset_attr_defs_from_attributes_node(
    attrs_node: dict[str, Any],
) -> list[DatasetAttributeComponentDef] | None:
    raw_defs = attrs_node.get("dataSet", attrs_node.get("dataset"))
    if not isinstance(raw_defs, list):
        return None
    out: list[DatasetAttributeComponentDef] = []
    for item in raw_defs:
        if not isinstance(item, dict):
            continue
        try:
            out.append(DatasetAttributeComponentDef.model_validate(item))
        except ValidationError:
            continue
    return out


def _structure_dict_for_dataset_attrs(
    payload: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any] | None:
    structures = data.get("structures")
    if isinstance(structures, list) and structures and isinstance(structures[0], dict):
        return structures[0]
    structure = payload.get("structure")
    if isinstance(structure, dict):
        return structure
    return None


def _infer_missing_dataset_attr_index(attr_def: DatasetAttributeComponentDef) -> int | None:
    if len(attr_def.values) == 1:
        return 0
    return None


def _resolve_dataset_attr_value(
    attr_def: DatasetAttributeComponentDef, raw_value_index: object
) -> str | None:
    if raw_value_index is None:
        return None
    if isinstance(raw_value_index, str):
        return raw_value_index
    if isinstance(raw_value_index, list):
        return ", ".join(str(v) for v in raw_value_index if v is not None) or None
    if not isinstance(raw_value_index, int):
        return str(raw_value_index)

    values = attr_def.values
    if raw_value_index >= len(values):
        return None

    value = values[raw_value_index]
    if isinstance(value, dict):
        for key in ("id", "name", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        return None
    return str(value)


def parse_dataset_level_attributes_map(payload: dict[str, Any]) -> dict[str, str | None]:
    """Parse dataset-level attribute ids to a single display string (or ``None``)."""
    inner = payload.get("data", payload)
    if not isinstance(inner, dict):
        return {}

    try:
        data_part = DatasetLevelAttributesData.model_validate(inner)
    except ValidationError:
        return {}

    if len(data_part.data_sets) != 1:
        return {}

    indices = data_part.data_sets[0].attributes
    if not isinstance(indices, list):
        return {}

    structure_dict = _structure_dict_for_dataset_attrs(payload, inner)
    if not structure_dict:
        return {}

    attrs_node = structure_dict.get("attributes")
    if not isinstance(attrs_node, dict):
        return {}

    attr_defs = _dataset_attr_defs_from_attributes_node(attrs_node)
    if attr_defs is None:
        return {}

    result: dict[str, str | None] = {}
    for idx, attr_def in enumerate(attr_defs):
        if idx < len(indices):
            raw_value_index = indices[idx]
        else:
            raw_value_index = _infer_missing_dataset_attr_index(attr_def)
            if raw_value_index is None:
                continue

        result[attr_def.id] = _resolve_dataset_attr_value(attr_def, raw_value_index)
    return result

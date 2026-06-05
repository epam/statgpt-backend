"""SDMX-JSON data message reader supporting both proxy and standard SDMX 3.0 APIs.

Both the ``dimensionGroup`` attribute category and the ``dimensionGroupAttributes``
data container are part of the standard SDMX-JSON 2.0.0 data schema — the schema
permits the attribute levels ``dataSet``, ``dimensionGroup``, ``series`` and
``observation``, and a ``dataSet`` may carry ``attributes``,
``dimensionGroupAttributes``, ``series`` and ``observations``.  The reader
therefore supports both variants seen in practice:

1. **Proxy SDMX-JSON** (e.g. IMF proxy API):

   - Group attributes use the ``dimensionGroup`` category and their values are
     carried in ``dataSets[].dimensionGroupAttributes``, keyed by a partial
     dimension key (e.g. ``"0:0::"`` for COUNTRY+INDICATOR, ``":0::"`` for
     INDICATOR only).  They are **not** indexed in the series-level data
     attribute arrays — only ``series``-level attributes are.
   - Observation dimension values may use the uncoded ``{"value": "2026"}`` syntax.
   - Uncoded attribute values (e.g. ``COUNTRY_UPDATE_DATE``) may appear as raw
     strings directly in the data attribute arrays rather than integer indices.

2. **Standard / alternative SDMX-JSON 2.0.0** (some SDMX 3.0 servers):

   - Servers may omit ``dimensionGroup`` and instead list those attributes under
     ``series`` with their ``relationship.dimensions`` metadata preserved.  The
     per-series data attribute arrays then carry positions for **all** series
     attributes, producing longer arrays with many ``null`` entries for
     attributes whose coded values don't apply to the given series.
   - Dimension values always use the coded ``{"id": "2026", "name": "2026"}``
     syntax and may include ``start``/``end`` timestamps.
   - The level key ``dataset`` (lowercase) may appear in place of ``dataSet``.
   - Attribute/dimension value objects may carry additional ``name`` and
     ``description`` fields alongside ``id``.

The reader normalises both variants so that downstream consumers receive
identical :class:`~sdmx.model.v21.DataSet` objects regardless of which API
produced the JSON.

References:
    - SDMX-JSON 2.0.0 data schema:
      https://json.sdmx.org/2.0.0/sdmx-json-data-schema.json
    - SDMX-JSON specification repository:
      https://github.com/sdmx-twg/sdmx-json
"""

import json
import logging
from typing import Any, cast
from warnings import warn

from dateutil.parser import isoparse
from sdmx.message import DataMessage, Header
from sdmx.model import common
from sdmx.model.common import Concept
from sdmx.model.v21 import (
    ActionType,
    AllDimensions,
    AttributeValue,
    Code,
    DataflowDefinition,
    DataSet,
    KeyValue,
)
from sdmx.reader.json import Reader, _org

log = logging.getLogger(__name__)

_LEVEL_ALIASES: dict[str, str] = {"dataset": "dataSet"}
"""Map non-canonical level names to their canonical form.

The official SDMX-JSON 2.0.0 schema uses camelCase ``dataSet`` for both
dimensions and attributes, but some implementations (e.g. the standard
SDMX 3.0 API) emit lowercase ``dataset`` instead.  This mapping normalises
the level name so that internal look-ups (e.g. :meth:`Reader._make_key` with
``"dataSet"``) work regardless of casing.
"""

_CODE_ACCEPTED_KEYS: frozenset[str] = frozenset({"id", "name", "description"})
"""Keys from an SDMX-JSON value object that are safe to forward to
:class:`~sdmx.model.v21.Code`.  Other keys present in the JSON
(``start``, ``end``, ``order``, ``parent``, ``links``, ``annotations``)
are not accepted by :class:`Code` and would raise :exc:`TypeError`.
"""


class StatGptSdmxProxyDataReader(Reader):
    """Read SDMX-JSON and expose it as instances from :mod:`sdmx.model`.

    Handles group attributes carried in ``dimensionGroupAttributes`` (proxy
    SDMX-JSON, using the standard ``dimensionGroup`` category) as well as the
    variant where servers fold those attributes under ``series`` instead.

    See the module docstring for a detailed comparison of both formats.
    """

    @classmethod
    def detect(cls, content: bytes | None) -> bool:
        warn(
            "Reader.detect(bytes); use Converter.handles() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        prefix = cls.binary_content_startswith
        if content is None or prefix is None:
            return False
        return content.startswith(prefix)

    def convert(self, data: Any, structure: Any = None, **kwargs: Any) -> DataMessage:
        msg = DataMessage()

        dsd = self._handle_deprecated_kwarg(structure, kwargs)
        if dsd:  # pragma: no cover
            if msg.dataflow is None:
                msg.dataflow = DataflowDefinition()
            cast(Any, msg.dataflow).structure = dsd

        data.default_size = -1
        tree = json.load(data)

        # Read the header — SDMX-JSON 1.0 uses "header", 2.0 uses "meta"
        try:
            elem = tree["header"]
        except KeyError:
            elem = tree["meta"]

        msg.header = Header(
            id=elem["id"],
            prepared=isoparse(elem["prepared"]),
            sender=_org(elem["sender"], cls=common.Agency),
        )

        # Locate the structure definition
        try:
            structure = tree["structure"]
        except KeyError:
            structure = tree["data"]["structures"][0]

        # -----------------------------------------------------------------
        # Read dimensions and values
        # -----------------------------------------------------------------
        # Both formats share the same dimension layout (``series`` and
        # ``observation``).  The standard format additionally includes a
        # ``dataset`` level (typically empty).  Level names are normalised
        # via ``_LEVEL_ALIASES`` so that ``_make_key("dataSet")`` works
        # regardless of whether the JSON used ``dataset`` or ``dataSet``.
        # -----------------------------------------------------------------
        self._dim_level: dict[Any, str] = {}
        self._dim_values: dict[Any, list[KeyValue]] = {}
        for level_name, level in structure["dimensions"].items():
            canonical = _LEVEL_ALIASES.get(level_name, level_name)
            for elem in level:
                d = msg.structure.dimensions.getdefault(
                    id=elem["id"], order=elem.get("keyPosition", -1)
                )

                self._dim_level[d] = canonical

                self._dim_values[d] = []
                for value in elem.get("values", []):
                    # Per SDMX-JSON ``dimValues`` schema, coded dimension
                    # values carry ``"id"`` while uncoded values (e.g. time
                    # periods in the proxy format) use ``"value"`` instead.
                    dim_value = value.get("id") or value.get("value")
                    self._dim_values[d].append(KeyValue(id=d.id, value=dim_value))

        for d in msg.structure.dimensions:
            if d.order == -1:
                d.order = len(msg.structure.dimensions)

        if all(level == "observation" for level in self._dim_level.values()):
            dim_at_obs = AllDimensions
        else:
            dim_at_obs = [dim for dim, level in self._dim_level.items() if level == "observation"]

        msg.observation_dimension = dim_at_obs

        # -----------------------------------------------------------------
        # Read attributes and values
        # -----------------------------------------------------------------
        # The two formats differ significantly in how attributes are
        # categorised (see module docstring).  Crucially:
        #
        # * **Proxy format** — ``dimensionGroup`` attributes appear in the
        #   structure but are NOT indexed in the series data arrays.  The
        #   data arrays only contain positions for ``series``-level attrs.
        #
        # * **Standard format** — there is no ``dimensionGroup``.  All
        #   dimension-related attributes are placed under ``series``, and
        #   the data arrays contain positions for ALL of them (with
        #   ``null`` for attributes that don't apply).
        #
        # Because ``_make_attrs`` filters by canonical level name, both
        # layouts produce correct alignment: the proxy's 4-element array
        # matches 4 ``series`` attrs, while the standard's 41-element
        # array matches 41 ``series`` attrs.
        # -----------------------------------------------------------------
        self._attr_level: dict[Any, str] = {}
        self._attr_values: dict[Any, list[AttributeValue]] = {}
        for level_name, level in structure["attributes"].items():
            canonical = _LEVEL_ALIASES.get(level_name, level_name)
            for attr in level:
                da = msg.structure.attributes.getdefault(
                    id=attr["id"],
                    concept_identity=Concept(name=attr.get("name", attr["id"])),
                )

                values = []
                for v in attr.get("values", []):
                    values.append(
                        AttributeValue(
                            value=(
                                _code_from_value(v) if "id" in v else v.get("name", v.get("value"))
                            ),
                            value_for=da,
                        )
                    )

                self._attr_level[da] = canonical

                if not len(values):
                    log.debug(f"No AttributeValues for attribute {repr(da)}")
                self._attr_values[da] = values

        self.msg = msg

        ds_key = self._make_key("dataSet")

        for ds in tree["data"]["dataSets"]:
            msg.data.append(self.read_dataset(ds, ds_key))

        return msg

    def read_dataset(self, root: dict[str, Any], ds_key: Any) -> DataSet:
        ds = DataSet(
            action=ActionType[root["action"].lower()],
            valid_from=root.get("validFrom", None),
        )
        ds.attrib.update(self._make_dataset_level_attrs(root.get("attributes", [])))

        # Proxy format only: attributes grouped by a subset of the series
        # dimensions (e.g. per INDICATOR or per COUNTRY+INDICATOR).  Each group
        # is applied to every series whose key matches the group's partial key.
        dim_group_attrs = self._make_dimension_group_attrs(root.get("dimensionGroupAttributes", {}))

        # Process series
        for key_values, elem in root.get("series", {}).items():
            series_key = self._make_key("series", key_values, base=ds_key)
            series_key.attrib = self._make_attrs("series", elem.get("attributes", []))
            for dim_filter, attrs in dim_group_attrs:
                if self._series_matches_dimension_group(series_key, dim_filter):
                    series_key.attrib.update(attrs)
            ds.add_obs(self.read_obs(elem, series_key=series_key), series_key)

        # Process bare observations
        ds.add_obs(self.read_obs(root, base_key=ds_key))

        return ds

    def _make_attrs(self, level: str, values: list[Any]) -> dict[str, AttributeValue]:
        """Resolve a data attribute array into a dict of :class:`AttributeValue`.

        Parameters
        ----------
        level:
            Canonical level name — ``"dataSet"``, ``"series"``, or
            ``"observation"``.  Only attributes whose recorded level matches
            are considered.  In the proxy format the ``"dimensionGroup"``
            attributes are stored separately and are **not** matched here,
            which is correct because the proxy's data arrays don't include
            positions for them.  In the standard format there is no
            ``dimensionGroup``; all relevant attributes are under ``"series"``
            and the arrays carry positions for every one of them.
        values:
            The raw attribute array from the JSON.  Each element is either an
            integer index into the attribute's coded value list, ``null``
            (meaning "not applicable"), or — in the proxy format — a raw
            string for uncoded attributes.
        """
        attrs = [a for a in self.msg.structure.attributes if self._attr_level[a] == level]
        result = {}
        for index, attr in zip(values, attrs):
            if index is None:
                continue
            if not isinstance(index, int):
                av = self._resolve_dataset_level_slot(attr, index)
                if av is not None and av.value is not None:
                    result[attr.id] = av
                continue
            if attr not in self._attr_values:
                log.debug(
                    "Attribute %s has no values; skip index %s",
                    attr.id,
                    index,
                )
                continue
            if index >= len(self._attr_values[attr]):
                log.warning(
                    "Attribute %s index %s out of range (%s)",
                    attr.id,
                    index,
                    len(self._attr_values[attr]),
                )
                continue
            av = self._attr_values[attr][index]
            result[av.value_for.id] = av
        return result

    def _make_dataset_level_attrs(self, values: list[Any]) -> dict[str, AttributeValue]:
        """Resolve ``dataSets[].attributes`` for components at the ``dataSet`` level.

        Unlike :meth:`_make_attrs` for ``series``/``observation``, supports ``null``,
        out-of-band strings, inline lists (e.g. localized text), integer indices into
        coded value lists (including empty lists), and implicit indices when the JSON
        omits trailing positions but a component has a single coded value.
        """
        attrs = [a for a in self.msg.structure.attributes if self._attr_level[a] == "dataSet"]
        result: dict[str, AttributeValue] = {}
        for idx, attr in enumerate(attrs):
            if idx < len(values):
                raw: Any = values[idx]
            else:
                coded = self._attr_values.get(attr, [])
                raw = 0 if len(coded) == 1 else None
                if raw is None:
                    continue

            av = self._resolve_dataset_level_slot(attr, raw)
            if av is not None:
                result[attr.id] = av
        return result

    def _resolve_dataset_level_slot(self, attr: Any, raw: Any) -> AttributeValue | None:
        if raw is None:
            return AttributeValue(value=None, value_for=attr)  # type: ignore[arg-type]

        if isinstance(raw, str):
            return AttributeValue(value=raw, value_for=attr)

        if isinstance(raw, list):
            joined = _inline_attr_text(raw)
            if joined is None:
                return AttributeValue(value=None, value_for=attr)  # type: ignore[arg-type]
            return AttributeValue(value=joined, value_for=attr)

        if not isinstance(raw, int):
            return AttributeValue(value=str(raw), value_for=attr)

        coded = self._attr_values.get(attr, [])
        if not len(coded) or raw >= len(coded):
            return AttributeValue(value=None, value_for=attr)  # type: ignore[arg-type]
        return coded[raw]

    def _make_dimension_group_attrs(
        self, raw: dict[str, list[Any]]
    ) -> list[tuple[dict[str, str | None], dict[str, AttributeValue]]]:
        """Resolve ``dataSets[].dimensionGroupAttributes`` into ``(filter, attrs)`` pairs.

        Group attributes (SDMX-JSON 2.0.0 ``dimensionGroup`` category) are attached
        to a *subset* of the series dimensions.  Each entry is keyed by a partial
        dimension key (e.g. ``"0:0::"`` for the COUNTRY+INDICATOR group, ``":0::"``
        for the INDICATOR-only group); empty segments are wildcards.  The value
        array aligns positionally to the ``dimensionGroup`` attributes declared in
        the structure, using the same encoding as :meth:`_resolve_dataset_level_slot`
        (``null``, an integer index into the coded value list, or an inline list).

        Returns ``(dim_filter, attrs)`` pairs where ``dim_filter`` maps a dimension
        id to the required dimension value for each non-wildcard position, and
        ``attrs`` is the resolved attribute dict applied to every matching series.
        """
        if not raw:
            return []

        dg_attrs = [
            a for a in self.msg.structure.attributes if self._attr_level[a] == "dimensionGroup"
        ]
        dims_ordered = sorted(self.msg.structure.dimensions, key=lambda d: d.order)

        resolved: list[tuple[dict[str, str | None], dict[str, AttributeValue]]] = []
        for partial_key, values in raw.items():
            dim_filter = self._dimension_group_filter(partial_key, dims_ordered)
            attrs: dict[str, AttributeValue] = {}
            for slot, attr in zip(values, dg_attrs):
                if slot is None:
                    continue
                av = self._resolve_dataset_level_slot(attr, slot)
                if av is not None and av.value is not None:
                    attrs[attr.id] = av
            if attrs:
                resolved.append((dim_filter, attrs))
        return resolved

    def _dimension_group_filter(
        self, partial_key: str, dims_ordered: list[Any]
    ) -> dict[str, str | None]:
        """Map a partial dimension key (``"0:0::"``) to ``{dim_id: dim_value}``.

        Each colon-separated segment is an index into the dimension's value list
        at the matching ``keyPosition``; empty segments are wildcards and omitted.
        """
        dim_filter: dict[str, str | None] = {}
        for position, segment in enumerate(partial_key.split(":")):
            if segment == "" or position >= len(dims_ordered):
                continue
            try:
                index = int(segment)
            except ValueError:
                continue
            dim = dims_ordered[position]
            values = self._dim_values.get(dim, [])
            if 0 <= index < len(values):
                dim_filter[dim.id] = values[index].value
        return dim_filter

    @staticmethod
    def _series_matches_dimension_group(series_key: Any, dim_filter: dict[str, str | None]) -> bool:
        """True when the series key matches the group filter on every constrained dimension."""
        series_values = {dim_id: kv.value for dim_id, kv in series_key.values.items()}
        return all(series_values.get(dim_id) == value for dim_id, value in dim_filter.items())


def _inline_attr_text(values: list[Any]) -> str | None:
    """Render an inline (uncoded) attribute value list to a display string.

    The proxy SDMX-JSON format stores uncoded attribute values inline as a list
    whose elements are plain scalars or localized-text objects
    (e.g. ``{"en": "Nominal GDP"}``).  Localized objects are unwrapped to their
    first available localization.  Returns ``None`` when nothing renders.
    """
    parts: list[str] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, dict):
            text = next((str(v) for v in item.values() if v is not None), None)
            if text is not None:
                parts.append(text)
        else:
            parts.append(str(item))
    return ", ".join(parts) or None


def _code_from_value(v: dict) -> Code:
    """Create a :class:`Code` from an SDMX-JSON value object.

    SDMX-JSON value objects may carry keys beyond what :class:`Code` accepts
    (e.g. ``start``, ``end``, ``order``, ``links``).  Only the keys listed in
    :data:`_CODE_ACCEPTED_KEYS` are forwarded to avoid :exc:`TypeError`.

    In the proxy format, values are typically minimal (``{"id": "9"}``).  In
    the standard SDMX-JSON 2.0.0 format, they often include ``name`` and
    ``description`` as well (``{"id": "9", "name": "Billions", ...}``).
    """
    return Code(**{k: v[k] for k in v if k in _CODE_ACCEPTED_KEYS})

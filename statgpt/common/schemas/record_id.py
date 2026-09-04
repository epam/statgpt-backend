"""Stable, opaque record identifiers for SDMX series returned by the data query tool.

A record identifier names a queried SDMX record so a follow-up MCP call can reference the
exact record an earlier call returned, without having to re-identify it by label (labels are
not unique across providers and change between vintages).

Format
------
``{agency}:{resource}({version})/{series_key}``

- ``{agency}:{resource}({version})`` is the dataflow reference (SDMX short URN): the
  maintaining agency (provider), the dataflow id, and the dataflow version.
- ``{series_key}`` is the SDMX 2.1 REST series key: the non-time dimension values in DSD
  order, ``'.'`` between dimensions, ``'+'`` within a dimension, and an empty segment for a
  dimension left unfiltered (a wildcard slot).

Example: ``IMF.RES:ED(1.0.0)/A.USD.`` selects the annual (``A``), USD (``USD``) series of the
``IMF.RES:ED(1.0.0)`` dataflow, with the third dimension wildcarded.

Stability
---------
The identifier is a pure function of the dataflow reference and the series key, so the same
logical record keeps the same identifier across sessions and server instances. It is
deterministic and carries no ephemeral (session- or request-scoped) state.

A change is only expected when the underlying record actually changes:
- a new dataflow version bumps ``(version)``;
- a change to the DSD (adding, removing, or reordering non-time dimensions) changes the key
  layout.

Both are breaking changes to the identifier and should be treated as such.

Opacity
-------
Callers (including the model) must treat the identifier as opaque and pass it through
verbatim. The composition is documented so its stability can be reasoned about, not so it can
be assembled or edited by hand.
"""

import re
from dataclasses import dataclass

from .query import JsonComponentQuery, JsonQueryOperator, JsonQueryWithMetadata

_KEY_SEPARATOR = "/"
_DIMENSION_SEPARATOR = "."
_VALUE_SEPARATOR = "+"

# Matches the dataflow-reference part of a record id, i.e. `AGENCY:RESOURCE(VERSION)`.
# Mirrors `JsonQuery._URN_PATTERN`: the same shape a query's `urn` is validated against.
_DATAFLOW_REF_PATTERN = re.compile(
    r"^(?P<agency>[A-Za-z0-9_.-]+):(?P<resource>[A-Za-z0-9_.@-]+)"
    r"\((?P<version>[A-Za-z0-9_.-]+)\)$"
)


def build_sdmx_series_key(
    filters: list[JsonComponentQuery],
    time_component: str,
    key_dimension_ids_in_dsd_order: list[str] | None,
) -> str:
    """Build the SDMX REST series key string from categorical filters.

    When ``key_dimension_ids_in_dsd_order`` is set (DSD order, excluding time), the key
    matches SDMX 2.1 expectations: ``'.'`` between dimensions, ``'+'`` within a dimension,
    and ``''`` for dimensions with no filter (wildcard slot). This ordering is what makes the
    key stable: it does not depend on the order the filters happen to be listed in.

    If that hint is absent, key segments follow the filter list order (legacy) and the time
    dimension is skipped.
    """
    if key_dimension_ids_in_dsd_order:
        by_id = {f.component_code: f for f in filters}
        parts: list[str] = []
        for dim_id in key_dimension_ids_in_dsd_order:
            if dim_id == time_component:
                continue
            f = by_id.get(dim_id)
            parts.append(_VALUE_SEPARATOR.join(f.values) if f is not None else "")
        return _DIMENSION_SEPARATOR.join(parts)
    parts = []
    for f in filters:
        if f.component_code == time_component:
            continue
        parts.append(_VALUE_SEPARATOR.join(f.values))
    return _DIMENSION_SEPARATOR.join(parts)


@dataclass(frozen=True)
class RecordId:
    """The parsed components of a record identifier.

    See the module docstring for the identifier format, stability guarantees, and opacity
    contract.
    """

    agency_id: str
    resource_id: str
    version: str
    series_key: str

    @property
    def dataflow_ref(self) -> str:
        """The dataflow reference (SDMX short URN): ``AGENCY:RESOURCE(VERSION)``."""
        return f"{self.agency_id}:{self.resource_id}({self.version})"

    def compose(self) -> str:
        """Render the identifier string."""
        return f"{self.dataflow_ref}{_KEY_SEPARATOR}{self.series_key}"

    @classmethod
    def parse(cls, value: str) -> "RecordId":
        """Parse a record identifier string back into its components.

        The dataflow reference cannot contain ``'/'``, so splitting on the first one
        separates it from the series key unambiguously.
        """
        dataflow_ref, separator, series_key = value.partition(_KEY_SEPARATOR)
        if not separator:
            raise ValueError(
                f"Invalid record id {value!r}: missing '{_KEY_SEPARATOR}' separating the "
                "dataflow reference from the series key."
            )
        match = _DATAFLOW_REF_PATTERN.match(dataflow_ref)
        if match is None:
            raise ValueError(
                f"Invalid record id {value!r}: dataflow reference must match "
                "'AGENCY:RESOURCE(VERSION)'."
            )
        return cls(
            agency_id=match["agency"],
            resource_id=match["resource"],
            version=match["version"],
            series_key=series_key,
        )

    def to_component_filters(
        self, key_dimension_ids_in_dsd_order: list[str]
    ) -> list[JsonComponentQuery]:
        """Reconstruct the categorical filters the series key encodes.

        Pairs each key segment with its dimension id (in DSD order) and emits an ``in`` filter
        per non-empty segment. Empty segments (wildcard slots) contribute no filter. This is
        the inverse of :func:`build_sdmx_series_key` for keys built from ``in`` filters, which
        is what the data query tool produces.
        """
        segments = self.series_key.split(_DIMENSION_SEPARATOR) if self.series_key else []
        filters: list[JsonComponentQuery] = []
        for dim_id, segment in zip(key_dimension_ids_in_dsd_order, segments):
            if not segment:
                continue
            filters.append(
                JsonComponentQuery(
                    component_code=dim_id,
                    operator=JsonQueryOperator.IN,
                    values=segment.split(_VALUE_SEPARATOR),
                )
            )
        return filters


def record_id_of(query: JsonQueryWithMetadata) -> RecordId:
    """Build the :class:`RecordId` for a single queried record."""
    series_key = build_sdmx_series_key(
        query.filters,
        query.metadata.time_period_dimension,
        query.metadata.key_dimension_ids_in_dsd_order,
    )
    return RecordId(
        agency_id=query.agency_id,
        resource_id=query.resource_id,
        version=query.version,
        series_key=series_key,
    )


def compose_record_id(query: JsonQueryWithMetadata) -> str:
    """Compose the stable record identifier string for a single queried record."""
    return record_id_of(query).compose()

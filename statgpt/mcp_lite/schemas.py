from typing import Literal

from pydantic import BaseModel, Field


class GlossaryTermPreview(BaseModel):
    """Short preview of a glossary term — enough for listing, not for use."""

    term: str
    domain: str


class GlossaryTermFull(BaseModel):
    """Full glossary term with definition and source."""

    term: str
    definition: str
    domain: str
    source: str


class GlossaryTerms(BaseModel):
    terms: list[GlossaryTermPreview] = Field(default_factory=list)


class DatasetSummary(BaseModel):
    """One row in `list_datasets`."""

    id: str = Field(description="Stable dataset identifier (the source-system id, e.g. 'BIS_CPI').")
    name: str = Field(description="Human-readable dataset title.")
    url: str | None = Field(default=None, description="Source page / data-explorer URL, if any.")


class Datasets(BaseModel):
    datasets: list[DatasetSummary] = Field(default_factory=list)


DimensionType = Literal["indicator", "non_indicator", "special", "time"]


class DimensionInfo(BaseModel):
    """One dimension in `dataset_structure`."""

    id: str
    name: str
    type: DimensionType
    alias: str | None = None
    codelist_size: int | None = Field(
        default=None,
        description="Number of available values. None for time dimensions.",
    )


class DatasetStructure(BaseModel):
    id: str = Field(description="Echoed dataset id.")
    name: str
    dims: list[DimensionInfo] = Field(default_factory=list)


class DimValue(BaseModel):
    """One value (code) of a categorical dimension."""

    code: str = Field(description="Source-system value id (e.g. 'DE' for Germany).")
    name: str = Field(description="Human-readable label.")


class DimValuesSample(BaseModel):
    dataset_id: str
    dim_id: str
    total: int = Field(description="Total number of available values for the dimension.")
    returned: int = Field(description="Number of values in `values` (<= total).")
    is_full: bool = Field(description="True when `values` contains every available value.")
    values: list[DimValue] = Field(default_factory=list)


class TimePeriod(BaseModel):
    start: str | None = Field(
        default=None,
        description="Earliest available period (SDMX time-period string, e.g. '2010-Q1').",
    )
    end: str | None = Field(
        default=None,
        description="Latest available period.",
    )


class AvailabilityResult(BaseModel):
    """Result of an availability query: which dim values are reachable under the filter."""

    dataset_id: str
    filter: dict[str, list[str]] = Field(
        description="Echo of the input `filter`.",
        default_factory=dict,
    )
    available: dict[str, list[DimValue]] = Field(
        description=(
            "Reachable values per dimension under the filter. "
            "Keys are dim ids; values are lists of {code, name}."
        ),
        default_factory=dict,
    )
    time_period: TimePeriod | None = Field(
        default=None,
        description="Min/max time period available under the filter, if exposed by the dataset.",
    )


class IndicatorMatch(BaseModel):
    """One match from `search_indicators` — a compound indicator that identifies a series."""

    code: str = Field(
        description=(
            "Composite indicator id — dot-joined values of the indicator dimensions "
            "(e.g. 'S12.P_F3_P_USD.A.S12'). Same shape SDMX uses for the data key."
        )
    )
    name: str = Field(description="Human-readable indicator name.")
    score: float = Field(description="Hybrid lex+sem score; higher = more relevant.")
    dimensions: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Pinned dim values that compose this indicator (indicator_dim_id -> code). "
            "Pass to `execute_sdmx_query.selection` to fetch this series; partial-pin + "
            "wildcard the rest if you want a broader slice."
        ),
    )
    available_in: list[str] | None = Field(
        default=None,
        description=(
            "Other dataset_ids that contain this exact indicator. Populated only on "
            "cross-dataset search (when `dataset_id` arg was null) and when ≥2 datasets "
            "share the indicator. The enclosing group's `dataset_id` carries the "
            "top-scoring occurrence; `available_in` lists the rest."
        ),
    )


class DatasetIndicatorGroup(BaseModel):
    """All matching indicators from one dataset, ranked within the dataset."""

    dataset_id: str = Field(description="Source-system id of the dataset.")
    best_score: float = Field(
        description="Score of the best-ranked indicator in this dataset (== matches[0].score)."
    )
    matches: list[IndicatorMatch] = Field(default_factory=list)


class IndicatorSearchResult(BaseModel):
    query: str
    n_total_matches: int = Field(
        default=0,
        description="Total indicator matches across all datasets (sum of len(g.matches)).",
    )
    datasets: list[DatasetIndicatorGroup] = Field(
        default_factory=list,
        description=(
            "Matching indicators grouped by dataset, sorted by `best_score` desc. "
            "Inspect this list end-to-end: each entry is a distinct dataset that "
            "contains at least one indicator matching the query."
        ),
    )


CodeMatchSource = Literal["non_indicator", "special"]


class CodeMatch(BaseModel):
    """One match from `search_codes` — an atomic dim value (filter code)."""

    source: CodeMatchSource = Field(description="Which dim surface this match came from.")
    dataset_id: str = Field(description="Source-system id of the dataset.")
    dim_id: str = Field(description="Dim this code belongs to (e.g. 'COUNTRY').")
    code: str = Field(description="Dim value code (e.g. 'DE').")
    name: str = Field(description="Human-readable label.")
    score: float = Field(description="Similarity score in [0, 1]; higher = more relevant.")


class SearchCodesResult(BaseModel):
    query: str
    matches: list[CodeMatch] = Field(default_factory=list)


class ExecuteResult(BaseModel):
    """Result of executing an SDMX data query."""

    dataset_id: str
    query_url: str | None = Field(
        default=None,
        description="Resolved upstream URL the data was fetched from, if exposed by the provider.",
    )
    row_count: int = Field(description="Total number of observations returned by the upstream.")
    truncated: bool = Field(
        description="True when only a preview of `data` is returned (row_count > len(data))."
    )
    time_range_actual: TimePeriod | None = Field(
        default=None,
        description="Earliest and latest period actually observed in the returned data.",
    )
    data: list[dict] = Field(
        default_factory=list,
        description=(
            "Observation rows. Each row is a flat dict with dimension codes and a 'value' key. "
            "Capped at the request's `limit` to keep payloads small."
        ),
    )
    warning: str | None = Field(
        default=None,
        description=(
            "Set when the upstream returned a non-success status but the call did not "
            "blow up — typically means the selected dim combination has no data, or the "
            "proxy returned an empty body for an out-of-bounds query. `row_count` will "
            "be 0; the agent should consider retrying with a different filter or wider "
            "time period (e.g. via `availability_query`)."
        ),
    )

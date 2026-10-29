import datetime
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, Field, JsonValue

from statgpt.common.utils.misc import normalize_whitespace

from .base import BaseYamlModel, DbDefaultBase
from .enums import DiscoveryIndexingStatus, DiscoverySummaryStep, DiscoveryValidationStatus

NormalizedStr = Annotated[str, AfterValidator(normalize_whitespace)]
"""A free-text field whose whitespace is normalized on every write path.

Applied by construction rather than by each service remembering to call the helper.
"""


def _require_non_empty(value: str) -> str:
    if not value:
        raise ValueError("must not be empty")
    return value


NormalizedKey = Annotated[
    str, AfterValidator(normalize_whitespace), AfterValidator(_require_non_empty)
]
"""Half of the natural key: normalized, and never empty.

A record with no key cannot be matched against what a channel already holds, so every write
path has to refuse one. Enforcing it here means no service can forget to.

The order is load-bearing. Annotated metadata applies left to right, so the emptiness check
must follow the normalizer - reversed, `'   '` passes as a three-character string and is then
normalized to `''` on its way into the database.
"""

MAX_DATASET_ID_LENGTH = 31
"""What Excel allows in a sheet name, which is what a dataset id has to fit into."""

FORBIDDEN_DATASET_ID_CHARS = ":\\/?*[]"
"""What Excel refuses in a sheet name."""

_RESERVED_SHEET_NAME = "history"
"""The one name Excel keeps for itself, whatever the case."""


def _require_sheet_name_safe(value: str) -> str:
    """Refuse an id Excel could not name a sheet after.

    The reason lives with the type rather than with the workbook parser: the id is the only
    thing pairing a dataset row with its indicators sheet, so an id no sheet can be named
    after is a dataset whose indicators can never be uploaded - whichever path stored it.
    """
    if len(value) > MAX_DATASET_ID_LENGTH:
        raise ValueError(
            f"must be at most {MAX_DATASET_ID_LENGTH} characters, since it also names the"
            f" dataset's indicators sheet"
        )
    forbidden = sorted({char for char in value if char in FORBIDDEN_DATASET_ID_CHARS})
    if forbidden:
        # Quoted, so the punctuation of the sentence around them is not read as one more.
        listed = " ".join(f'"{char}"' for char in forbidden)
        raise ValueError(
            f"must not contain the characters {listed}, since it also names the dataset's"
            f" indicators sheet"
        )
    if value.startswith("'") or value.endswith("'"):
        raise ValueError(
            "must not start or end with an apostrophe, since it also names the dataset's"
            " indicators sheet"
        )
    if value.lower() == _RESERVED_SHEET_NAME:
        raise ValueError(
            f"must not be {value!r}, a sheet name Excel reserves, since it also names the"
            f" dataset's indicators sheet"
        )
    return value


DatasetIdKey = Annotated[NormalizedKey, AfterValidator(_require_sheet_name_safe)]
"""The dataset half of the natural key, which doubles as the name of its indicators sheet.

Order is load-bearing here too: the length is counted after normalization, so padding does
not spend the 31 characters Excel allows.
"""


class DiscoveryValidationIssue(BaseYamlModel):
    """One reason a discovery dataset record cannot be indexed."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    field: str
    """The record field the issue concerns."""

    message: str
    """What a submitter needs to fix, in plain English."""


class DiscoverySummaryCall(BaseYamlModel):
    """One LLM answer of an indicator summarization, rejected or accepted."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    step: DiscoverySummaryStep

    part: int | None = None
    """1-based number of the summarized part of the list; `None` for the merge."""

    answer: dict[str, JsonValue] | None = None
    """The answer as the model returned it: reasoning, units and groups.

    `None` when it was not a JSON object - cut short, refused or filtered; its text is not kept.
    """

    error: str | None = None
    """What was wrong with the answer, as sent back to the model; `None` for the accepted one."""


class DiscoveryDatasetBase(BaseYamlModel):
    """One dataset described for discovery, as a submitter fills it in.

    Columns A-I come from the workbook's `Datasets` sheet; `indicators_coverage` is assembled
    from the dataset's own indicators sheet, or supplied as a flat column by a CSV.

    Makes a dataset discoverable without onboarding it (Grade C): the agent can
    recognize the dataset exists and refer the user to its official source. Every
    field is free text and is stored as submitted apart from whitespace
    normalization; the semicolon-separated lists inside cells are not parsed.

    ``agency`` and ``dataset_id`` form the natural key together with the channel,
    compared ignoring case and surrounding or repeated whitespace.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    reference_area: NormalizedStr = ""
    """Column A. Countries covered, as English name + ISO 3166-1 alpha-3 code, ';'-separated.
    A group label such as 'Euro area' or 'World' is allowed."""

    regional_coverage: NormalizedStr = ""
    """Column B. Sub-national levels the data is broken down by; 'None' for national totals only."""

    agency: NormalizedKey
    """Column C. Official English name of the publisher + acronym. Half of the natural key."""

    dataset_id: DatasetIdKey
    """Column D. The source's own dataset identifier, verbatim. Half of the natural key.

    Also the name of the dataset's indicators sheet, which is what constrains its shape.
    """

    name: NormalizedStr = ""
    """Column E. The official dataset title, in English, as published."""

    description: NormalizedStr = ""
    """Column F. What is measured, how it is broken down, and whether values are adjusted."""

    url: NormalizedStr = ""
    """Column G. Deepest stable public URL for the dataset - what a referral links to."""

    time_coverage: NormalizedStr = ""
    """Column H. 'From A to B' at the dataset's own granularity."""

    frequency_coverage: NormalizedStr = ""
    """Column I. ';'-separated frequencies from the template's vocabulary."""

    indicators_coverage: NormalizedStr = ""
    """The dataset's complete indicator list, each entry with its units of measure in
    parentheses, ';'-separated.

    No column of the `Datasets` sheet: a workbook assembles it from the dataset's own
    indicators sheet, one row per indicator. A CSV carries it as this one flat column, which
    is also how the export writes it.
    """


class DiscoveryDataset(DiscoveryDatasetBase, DbDefaultBase):
    """A stored discovery dataset record, with its derived, validation and indexing state."""

    channel_id: int = Field(description="The ID of the channel this record belongs to.")

    dataset_id: NormalizedKey
    """Relaxed back to the plain key: reading a record must never fail on its own contents.

    Records stored before the indicators sheet existed can hold an id no sheet could be
    named after. The rule applies to what a write submits, so those rows keep being read,
    exported and published; only resubmitting such an id is refused.
    """

    indicators_summary: str = ""
    """What of `indicators_coverage` is published, derived by the indexing run. Read-only.

    The parsed list, with a units of measure shared by every indicator stated once as a
    `Unit: ...` prefix. A list longer than the channel's threshold is summarized by an LLM into
    indicator groups instead. It is a column of its own because the raw list is what a
    submitter authored and must not be overwritten.
    """

    indicators_summarized_at: datetime.datetime | None = None
    """When an LLM last produced `indicators_summary`. Read-only.

    `None` when the summary is a copy of the parsed list, or has not been derived yet.
    """

    indicators_summary_calls: list[DiscoverySummaryCall] | None = Field(
        default=None, exclude_if=lambda calls: calls is None
    )
    """Every LLM answer that went into `indicators_summary`, in order. Read-only.

    Rejected answers are included, so the retries are the entries with an `error`. `None`
    when the summary is a copy of the parsed list, has not been derived yet, or the calls
    were not asked for; omitted from the response then.
    """

    indicators_summary_failed_calls: list[DiscoverySummaryCall] | None = Field(
        default=None, exclude_if=lambda calls: calls is None
    )
    """The answers of the last summarization that failed, in order. Read-only.

    Empty when it failed before any answer, on a request error say. Cleared by the next
    summarization that succeeds. Omitted from the response unless asked for.
    """

    missing_indicators: str = ""
    """Indicators a user could expect but the dataset does not contain. Read-only.

    No longer submitted nor derived: the indexing run leaves it empty, clearing values
    authored under the old template.
    """

    validation_status: DiscoveryValidationStatus
    """Verdict of the last indexing job to evaluate this record."""

    validation_issues: list[DiscoveryValidationIssue] | None = None
    """Why the record is invalid. Non-empty exactly when `validation_status` is INVALID."""

    validated_at: datetime.datetime | None = None
    """When the record was last evaluated."""

    indexing_status: DiscoveryIndexingStatus
    """Whether the record is published to the discovery index."""

    indexed_at: datetime.datetime | None = None
    """When the record was last published."""

    index_error: str | None = None
    """Why publishing the record last failed."""


class DiscoveryDatasetUpdate(BaseYamlModel):
    """The fields of a discovery dataset record an admin may edit.

    A record cannot change channel, so there is no `channel_id` here. Omitted fields are
    left as they are; every supplied field is whitespace-normalized. To clear a field,
    send an empty string - `null` means "not provided".

    The two halves of the natural key are the exception: they can be corrected but not
    cleared, since a record with no key cannot be matched against a channel's contents.

    `indicators_summary`, its `indicators_summarized_at` and summary calls, and
    `missing_indicators` are absent because the indexing run owns them; an edit that could set
    them would be overwritten by the next run.
    """

    reference_area: NormalizedStr | None = None
    regional_coverage: NormalizedStr | None = None
    agency: NormalizedKey | None = None
    dataset_id: DatasetIdKey | None = None
    name: NormalizedStr | None = None
    description: NormalizedStr | None = None
    url: NormalizedStr | None = None
    time_coverage: NormalizedStr | None = None
    frequency_coverage: NormalizedStr | None = None
    indicators_coverage: NormalizedStr | None = None


class DiscoveryDatasetUpdateBulk(DiscoveryDatasetUpdate):
    id: int = Field(description="The ID of the record to update.")


class DiscoveryDatasetStats(BaseYamlModel):
    """How many of a channel's discovery dataset records sit in each status and agency.

    The two status maps carry every member of their enum, zeros included, so a caller can
    render the whole breakdown without deciding what an absent key means. Agencies are not
    an enum, so `by_agency` carries only the ones the channel actually holds.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    total: int = 0
    """Records the channel holds."""

    by_validation_status: dict[DiscoveryValidationStatus, int] = Field(default_factory=dict)
    """Record count per validation verdict."""

    by_indexing_status: dict[DiscoveryIndexingStatus, int] = Field(default_factory=dict)
    """Record count per indexing state."""

    by_agency: dict[str, int] = Field(default_factory=dict)
    """Record count per agency, largest first, so the values double as filter arguments.

    Keys are spelled as stored, one per agency the `agency` filter can match: records whose
    agency differs only in case or spacing are one entry, since the filter folds them
    together too.
    """


class DiscoveryPayloadProblem(BaseYamlModel):
    """One structural problem that kept a payload from being saved."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    message: str
    """What is wrong, naming the offending value."""

    field: str | None = None
    """The record field the problem concerns, when it concerns one."""

    index: int | None = None
    """0-based position in a JSON payload."""

    row: int | None = None
    """1-based row number in the uploaded file."""

    cell: str | None = None
    """Cell reference in the uploaded file, e.g. 'D14'."""


class DiscoveryPayloadErrorDetail(BaseYamlModel):
    """The body of a 400 raised when a write is structurally unusable."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    message: str
    """What is wrong, naming how many problems were found when they can be itemized."""

    problems: list[DiscoveryPayloadProblem]
    """The problems, capped by the configured limit. Empty when the file was unreadable."""

    truncated: bool = False
    """True when more problems were found than are reported here."""


class DiscoveryPayloadErrorResponse(BaseYamlModel):
    """The 400 response body, declared so the per-cell report reaches the OpenAPI schema.

    FastAPI's own error shape carries a string in `detail`; this one carries an object, so a
    generated client only knows about it if the endpoints advertise this model.
    """

    model_config = ConfigDict(use_attribute_docstrings=True)

    detail: DiscoveryPayloadErrorDetail


class DiscoveryUploadSummary(BaseYamlModel):
    """What an upload did to the records a channel holds."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    created: int = 0
    """Records inserted."""

    updated: int = 0
    """Records that already existed and whose fields changed."""

    unchanged: int = 0
    """Records that already existed with identical fields. Not rewritten, so statuses are kept."""

    deleted: int = 0
    """Records absent from the file and removed. Always 0 in upsert mode."""

    rows_read: int = 0
    """Data rows found in the file, excluding the header and blank rows."""

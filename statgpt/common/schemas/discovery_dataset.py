import datetime
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, Field

from statgpt.common.utils.misc import normalize_whitespace

from .base import BaseYamlModel, DbDefaultBase
from .enums import DiscoveryIndexingStatus, DiscoveryValidationStatus

NormalizedStr = Annotated[str, AfterValidator(normalize_whitespace)]
"""A free-text field whose whitespace is normalized on every write path.

Applied by construction rather than by each service remembering to call the helper.
"""


class DiscoveryValidationIssue(BaseYamlModel):
    """One reason a discovery dataset record cannot be indexed."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    field: str
    """The record field the issue concerns."""

    message: str
    """What a submitter needs to fix, in plain English."""


class DiscoveryDatasetBase(BaseYamlModel):
    """One dataset described for discovery, as filled in on the Datasets sheet.

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

    excluded_regional_values: NormalizedStr = ""
    """Column C. Sub-national values a user could expect but the dataset does not contain."""

    agency: NormalizedStr
    """Column D. Official English name of the publisher + acronym. Half of the natural key."""

    dataset_id: NormalizedStr
    """Column E. The source's own dataset identifier, verbatim. Half of the natural key."""

    name: NormalizedStr = ""
    """Column F. The official dataset title, in English, as published."""

    description: NormalizedStr = ""
    """Column G. What is measured, how it is broken down, and whether values are adjusted."""

    url: NormalizedStr = ""
    """Column H. Deepest stable public URL for the dataset - what a referral links to."""

    time_coverage: NormalizedStr = ""
    """Column I. 'From A to B' at the dataset's own granularity."""

    frequency_coverage: NormalizedStr = ""
    """Column J. ';'-separated frequencies from the template's vocabulary."""

    indicators_coverage: NormalizedStr = ""
    """Column K. The indicators, ';'-separated, each with its units of measure in parentheses."""

    missing_indicators: NormalizedStr = ""
    """Column L. Indicators a user could expect but the dataset does not contain."""


class DiscoveryDataset(DiscoveryDatasetBase, DbDefaultBase):
    """A stored discovery dataset record, with its validation and indexing state."""

    channel_id: int = Field(description="The ID of the channel this record belongs to.")

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
    """

    reference_area: NormalizedStr | None = None
    regional_coverage: NormalizedStr | None = None
    excluded_regional_values: NormalizedStr | None = None
    agency: NormalizedStr | None = None
    dataset_id: NormalizedStr | None = None
    name: NormalizedStr | None = None
    description: NormalizedStr | None = None
    url: NormalizedStr | None = None
    time_coverage: NormalizedStr | None = None
    frequency_coverage: NormalizedStr | None = None
    indicators_coverage: NormalizedStr | None = None
    missing_indicators: NormalizedStr | None = None


class DiscoveryDatasetUpdateBulk(DiscoveryDatasetUpdate):
    id: int = Field(description="The ID of the record to update.")


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
    """Summary naming how many problems were found."""

    problems: list[DiscoveryPayloadProblem]
    """The problems, capped by the configured limit."""

    truncated: bool = False
    """True when more problems were found than are reported here."""


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

    rows_skipped: int = 0
    """Blank rows skipped, such as the empty formatted rows the template ships with."""

from __future__ import annotations

import datetime
import enum
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StrictStr, model_validator

from common.config import multiline_logger as logger
from common.schemas import RAGVersion
from common.utils import IntervalProcessor
from statgpt.schemas.file_rags import BaseRagState

# ~~~~~~~~~~~~~~~~~~~~~ Models for LLM ~~~~~~~~~~~~~~~~~~~~~


class SingleFilterLLMOutput(BaseModel):
    """Base class for single filter outputs from LLM Filter Builders."""


class TimePeriodFilter(SingleFilterLLMOutput):
    """LLM response model for prefilter time period filter."""

    start: str = Field(description='YYYY-MM-DD date', default='')
    end: str = Field(description='YYYY-MM-DD date', default='')
    date_format: ClassVar[str] = "%Y-%m-%d"

    def is_empty(self) -> bool:
        return self.start == '' and self.end == ''

    @classmethod
    def parse_date(cls, date_str: str) -> datetime.date | None:
        """Parse a date string to a date object. Return None if invalid."""
        try:
            return datetime.datetime.strptime(date_str, cls.date_format).date()
        except ValueError:
            return None

    def fix_llm_hallucinations(self) -> TimePeriodFilter:
        """
        LLM might occasionally set end or start date after today. Here we fix it.
        """
        today = datetime.date.today()
        parsed_start_date = self.parse_date(self.start)
        parsed_end_date = self.parse_date(self.end)
        if parsed_start_date and parsed_start_date > today:
            # if start is in the future, we consider filter to be invalid. return no time filter
            return TimePeriodFilter(start='', end='')
        if parsed_end_date and parsed_end_date >= today:
            # if end date is in the future,
            # or if end date is today, remove end filter
            # as it does no actual filtering
            # (we assume no publications have publish date in future).
            return TimePeriodFilter(start=self.start, end='')
        return self


class PublicationTypesFilter(SingleFilterLLMOutput):
    """Filter for publication types."""

    publication_types: list[str] = Field(
        description='List of requested publication types. Use empty list if not applicable.',
        default_factory=list,
    )


class LatestFilter(SingleFilterLLMOutput):
    """Filter for latest data."""

    is_latest: bool = Field(
        default=False, description="True if the time period is 'latest', False otherwise."
    )


class LastNPublicationsFilter(SingleFilterLLMOutput):
    """Filter for limiting the number of publications."""

    last_n_publications: int | None = Field(
        default=None,
        description="Limit the number of publications to N last publications.",
    )


class RagFilterLLMOutput(BaseModel):
    # NOTE: currently we do not allow to specify different time filters
    # for different publication types

    time_period: TimePeriodFilter = Field()
    publication_types: list[str] = Field(
        description='List of requested publication types. Use empty list if not applicable.'
    )
    last_n_publications: int | None = Field(
        default=None,
        description="Limit the number of publications to N last publications.",
    )
    is_latest: bool = Field(
        default=False,
        description="True if the time period is 'latest', False otherwise.",
    )

    def __str__(self):
        return f"'{{time_period': {{start: {self.time_period.start}, end: {self.time_period.end}}}, 'publication_types': {self.publication_types}, is_latest: {self.is_latest}, 'last_n_publications': {self.last_n_publications}}}"

    def is_empty(self) -> bool:
        return (
            not self.publication_types
            and self.time_period.is_empty()
            and self.is_latest is False
            and self.last_n_publications is None
        )


# ~~~~~~~~~~~~~~~~~~~~~ Models for Dial RAG ~~~~~~~~~~~~~~~~~~~~~


class TimePeriodFilterDial(BaseModel):
    start: datetime.date | None = Field()
    end: datetime.date | None = Field()

    @model_validator(mode='after')
    def check_dates(self) -> Self:
        if self.start is None and self.end is None:
            raise ValueError("Both start and end dates cannot be None.")

        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("Start date cannot be after end date.")

        return self

    @classmethod
    def decode_latest_value(
        cls,
        decoder: str,
        end_date: datetime.date | None,
        reference_date: datetime.date | None = None,
    ) -> Self:
        """
        get decoded "latest" time filter, using specified end date.

        reference_date may be passed to decode end_date to a specific date;
        e.g. it's used during evaluation to correctly evaluate past artifacts.
        if end_date is None, we need to use reference_date or today as end while decoding.
        """

        end_date_for_decoder = end_date or reference_date or datetime.date.today()

        interval_processor = IntervalProcessor()
        # interval processor operates on datetimes and not on dates
        end_datetime = datetime.datetime.combine(end_date_for_decoder, datetime.datetime.min.time())
        start_datetime, _ = interval_processor.get_interval(decoder, date=end_datetime)

        # use original end_date, since we don't want to change Nones there
        return cls(start=start_datetime.date(), end=end_date)

    def __str__(self):
        return f"{{start: {self.start}, end: {self.end}}}"


class RagFilterDialSingle(BaseModel):
    publication_date: TimePeriodFilterDial | None = Field()
    publication_type: str | None = Field()

    @model_validator(mode='after')
    def check_filter(self) -> Self:
        if self.publication_date is None and self.publication_type is None:
            raise ValueError("At least one of time_period or publication_type must be provided.")
        return self


@enum.unique
class SortOrder(enum.StrEnum):
    asc = "asc"
    desc = "desc"


class TopNDocuments(BaseModel):
    sort_by: list[
        Literal["publication_date"]  # Currently, only publication_date sorting is used by StatGPT
    ] = Field(
        default_factory=lambda: ["publication_date"],  # type: ignore
        min_length=1,
        description="List of metadata fields to sort documents by.",
    )
    order: SortOrder = Field(
        default=SortOrder.desc,
        description=f"Sorting order (`{SortOrder.asc.value}`: from low to high, `{SortOrder.desc.value}`: from high to low)",
    )
    limit: PositiveInt = Field(
        description="Maximum number of documents to use after sorting.",
    )

    def __str__(self) -> str:
        return f"sort_by: {self.sort_by}, order: {self.order.value}, limit: {self.limit}"


class RagFilterDial(BaseModel):
    """
    This model is based on RetrieverConfig model in Dial RAG.
    We use it to send requests to Dial RAG
    """

    filters: list[RagFilterDialSingle] = Field(default_factory=list)
    top_n: TopNDocuments | None = Field(
        default=None,
        description="If specified, search only within top N documents sorted by given fields after applying all filters.",
    )

    # TODO: add validator checking that each FilterSingle has unique publication type.
    # and if there is a FilterSingle with no publication type filter (global time filter),
    # then there are no other filters

    def __eq__(self, value) -> bool:
        # Validate value
        if value is None:
            return False
        if not isinstance(value, RagFilterDial):
            logger.warning("Comparison is only supported between RagFilterDial instances.")
            return NotImplemented
        # check filters
        if len(self.filters) != len(value.filters):
            return False
        for f in self.filters:
            if f not in value.filters:
                return False
        # check top_n
        if self.top_n != value.top_n:
            return False
        # everything is equal
        return True

    def __str__(self):
        date2types = {}
        for f in self.filters:
            pub_date_str = str(f.publication_date)
            date2types.setdefault(pub_date_str, []).append(f.publication_type)
        return (
            "\n".join(f"{date} -> {types}" for date, types in date2types.items())
            + "\n"
            + f"Last N Documents: {self.top_n.limit if self.top_n else 'None'}"
        )

    def as_dial_dict(self) -> dict:
        """
        convert to dict accepted by DIAL RAG
        """
        return self.model_dump(mode='json', exclude_none=True)


# ~~~ Models for Dial RAG Metadata ~~~


class DialRagMetadataResponse(BaseModel):
    class Dimension(BaseModel):
        name: StrictStr = Field()
        values: list[StrictStr] = Field()

    schema_: dict = Field(
        alias='schema',
        description="Currently not used. If you need this field, you must first define it meaningfully.",
    )
    dimensions: list[Dimension] = Field()

    model_config = ConfigDict(populate_by_name=True)


class DialRagMetadata(BaseModel):
    publication_dates: set[StrictStr] = Field()
    publication_types: set[StrictStr] = Field()

    @classmethod
    def from_response(cls, response: DialRagMetadataResponse) -> "DialRagMetadata":
        return cls(
            publication_dates=cls._extract_dimension_values(
                response.dimensions, "publication_date"
            ),
            publication_types=cls._extract_dimension_values(
                response.dimensions, "publication_type"
            ),
        )

    @staticmethod
    def _extract_dimension_values(
        metadata: list[DialRagMetadataResponse.Dimension], dim_name: str
    ) -> set[StrictStr]:
        for dim in metadata:
            if dim.name == dim_name:
                return set(dim.values)
        raise ValueError(f"Metadata doesn't contain '{dim_name}' dimension.")


# ~~~~~~~~~~~~~~~~~~~~~ Final Models ~~~~~~~~~~~~~~~~~~~~~


class PreFilterResponse(BaseModel):
    user_friendly_error: str | None = Field(default=None)
    detailed_error: str | None = Field(default=None)
    llm_output: RagFilterLLMOutput | None = Field(default=None)
    # NOTE: Currently None means no filter applied; but RagFilterDial can also contain an empty list;
    # NOTE: Should we replace all the None logic with just an empty list?
    rag_filter: RagFilterDial | None = Field(default=None)


class DialRagState(BaseRagState):
    version: RAGVersion = RAGVersion.DIAL
    pre_filter: PreFilterResponse
    metadata: DialRagMetadata | None
    prefilter_decoder_of_latest: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping the publication type to a function that generates a time range "
        "corresponding to the 'latest'",
    )
    attachments: list[dict[str, Any]] = Field(default_factory=list)

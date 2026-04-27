import logging
from abc import ABC
from collections.abc import Generator
from typing import Annotated, Literal, Self

from pydantic import Field, SerializeAsAny, StrictStr, model_validator

from statgpt.common.config.utils import replace_env

from .base import BaseModel
from .enums import DimensionType, SpecialNonIndicatorDimensions
from .indexing import IndexingField, IndexingHashMixin
from .query import Query

_log = logging.getLogger(__name__)


class VirtualDimensionValue(BaseModel):
    id: Annotated[str, IndexingField()]
    name: Annotated[str, IndexingField()]
    description: str | None


class VirtualDimensionConfig(BaseModel, IndexingHashMixin):
    name: Annotated[str, IndexingField()] = Field(description="The name of the virtual dimension")
    description: str | None = Field(description="The description of the virtual dimension")
    value: Annotated[VirtualDimensionValue, IndexingField()] = Field(
        description="The value of the virtual dimension"
    )


class ProviderAgency(BaseModel):
    id: StrictStr
    count: int
    name: StrictStr


class DatasetCitation(BaseModel):
    provider: StrictStr | None = Field(default=None)
    provider_template: StrictStr | None = Field(default=None)
    provider_agencies: list[ProviderAgency] | None = Field(default=None)
    last_updated: StrictStr | None = Field(default=None)
    url: StrictStr | None = Field(default=None)
    description: StrictStr | None = Field(default=None)

    def get_url(self) -> str | None:
        if self.url:
            return replace_env(self.url)
        return None

    @model_validator(mode='after')
    def _check_provider_template_and_provider_agencies(self) -> Self:
        if self.provider_template and not self.provider_agencies:
            raise ValueError("provider_template is present but provider_agencies is not")
        if not self.provider_template and self.provider_agencies:
            raise ValueError("provider_template is not present but provider_agencies is")
        return self

    @model_validator(mode='after')
    def _check_provider_agencies(self) -> Self:
        if not self.provider_agencies:
            return self

        if len(self.provider_agencies) == 1:
            raise ValueError(
                "provider_agencies field must not be used with only 1 agency - "
                "use 'provider' field instead"
            )

        ids = set([agency.id for agency in self.provider_agencies])
        if len(ids) != len(self.provider_agencies):
            raise ValueError("provider_agencies field must not contain duplicated ids")

        names = set([agency.name for agency in self.provider_agencies])
        if len(names) != len(self.provider_agencies):
            raise ValueError("provider_agencies field must not contain duplicated names")

        return self

    @property
    def provider_agency_names_with_fallback_to_provider(self) -> list[str] | None:
        if self.provider_agencies:
            return [agency.name for agency in self.provider_agencies]
        if self.provider:
            return [self.provider]
        return None


class IndexerIndicatorAnnotationConfig(BaseModel):
    description: Annotated[str, IndexingField()] = Field(
        description="annotation name to get indicator description", default=""
    )


class IndexerIndicatorConfig(BaseModel):
    unpack: Annotated[bool, IndexingField()] = Field(
        default=False,
        description=(
            "When True, LLM selects canonical 'primary' from similar indicators found via hybrid search. "
            "When False, primary is taken from first dimension name and normalized."
        ),
    )
    super_primary: Annotated[bool, IndexingField()] = Field(
        default=False,
        description=(
            "Only applies when unpack=False. "
            "When True, primary is concatenated from first 3 dimensions. "
            "When False, primary is taken from first dimension only."
        ),
    )
    annotations: Annotated[IndexerIndicatorAnnotationConfig | None, IndexingField()] = Field(
        default=None
    )


class IndexerConfig(BaseModel, IndexingHashMixin):
    indicator: Annotated[IndexerIndicatorConfig, IndexingField()] = Field(
        description="indicator_config", default_factory=IndexerIndicatorConfig
    )


class BaseDimensionConfig(BaseModel, IndexingHashMixin):
    dimension_type: Annotated[str | None, IndexingField()]
    alias: Annotated[str | None, IndexingField()] = Field(default=None)
    is_required: bool = Field(
        default=False,
        description=(
            "Whether this dimension is one of the required dimensions to build a dataset query. "
            "Dataset query MUST have non-empty dimension query "
            "to AT LEAST ONE of the required dimensions. "
            "Otherwise, dataset query is filtered out and is not executed."
        ),
    )
    virtual: Annotated[VirtualDimensionConfig | None, IndexingField()] = Field(
        default=None,
        description=(
            "If set, defines a virtual dimension to be added to the dataset. "
            "E.g., national agency datasets can't have Country dimension in the source, "
            "so we add it as a virtual dimension with a fixed value representing the country of the agency."
        ),
    )
    all_values: VirtualDimensionValue | None = Field(
        default=None,
        description=(
            "If set, defines a special 'All-values' item for this dimension"
            " which can be used to set '*' filter in the query"
        ),
    )
    default_queries: list[Query] | None = Field(default=None)

    @property
    def type(self) -> DimensionType:
        if self.dimension_type is None:
            raise ValueError("dimension_type is not set")
        return DimensionType(self.dimension_type)


class IndicatorDimensionConfig(BaseDimensionConfig):
    dimension_type: Annotated[Literal["INDICATOR"], IndexingField()] = "INDICATOR"


class SpecialDimensionConfig(BaseDimensionConfig):
    dimension_type: Annotated[Literal["SPECIAL"], IndexingField()] = "SPECIAL"
    processor_id: Annotated[str, IndexingField()] = Field(
        description=(
            "The ID of the processor to handle this special dimension. "
            "NOTE: processors are defined in the channel configuration."
        )
    )


class NonIndicatorDimensionConfig(BaseDimensionConfig):
    dimension_type: Annotated[Literal["NON_INDICATOR"], IndexingField()] = "NON_INDICATOR"
    subtype: Annotated[SpecialNonIndicatorDimensions | None, IndexingField()] = Field(default=None)
    # named_entity: str  # TODO


class TimePeriodDimensionConfig(BaseDimensionConfig):
    dimension_type: Annotated[Literal["TIME_PERIOD"], IndexingField()] = "TIME_PERIOD"


DIMENSION_CONFIG_TYPES = Annotated[
    (
        IndicatorDimensionConfig
        | SpecialDimensionConfig
        | NonIndicatorDimensionConfig
        | TimePeriodDimensionConfig
    ),
    Field(discriminator='dimension_type'),
]


class BaseDataSetConfig(BaseModel):
    is_official: bool = Field(default=False)
    citation: DatasetCitation | None = Field(default=None)
    view_in_data_explorer: bool = Field(
        default=True,
        description=(
            "Whether to include 'View data in explorer' links for this dataset in formatted query "
            "results when a query URL is available."
        ),
    )
    data_explorer_url: str | None = Field(
        default=None,
        description=(
            "Base URL of the data explorer UI for this dataset. When set, it overrides the "
            "data source default for explorer links in query results and for dataset_url when "
            "no citation URL is configured."
        ),
    )
    indexer: Annotated[IndexerConfig | None, IndexingField()] = Field(default=None)
    pinned_columns: list[str] = Field(
        description="Column names and order to pin in the data in grid", default_factory=list
    )

    def get_data_explorer_url(self) -> str | None:
        if self.data_explorer_url:
            return replace_env(self.data_explorer_url)
        return None


class DataSetConfigTemplate(BaseDataSetConfig):

    dimensions: dict[str, SerializeAsAny[BaseDimensionConfig]] = Field(
        description="The draft configuration of the each dimension in the dataset by its ID",
        default_factory=dict,
    )


class DataSetConfig(BaseDataSetConfig, ABC, IndexingHashMixin):

    dimensions: Annotated[dict[str, DIMENSION_CONFIG_TYPES], IndexingField()] = Field(
        description="The configuration of the each dimension in the dataset by its ID",
        default_factory=dict,
    )

    def get_dimension_aliases(self) -> dict[str, str]:
        return {
            dim_id: dim_config.alias
            for dim_id, dim_config in self.dimensions.items()
            if dim_config.alias
        }

    @property
    def special_dimensions(self) -> Generator[tuple[str, SpecialDimensionConfig], None, None]:
        for dim_id, dim_config in self.dimensions.items():
            if dim_config.type is DimensionType.SPECIAL:
                yield dim_id, dim_config  # type: ignore[misc]

    @property
    def non_indicator_dimensions(
        self,
    ) -> Generator[tuple[str, NonIndicatorDimensionConfig], None, None]:
        for dim_id, dim_config in self.dimensions.items():
            if dim_config.type is DimensionType.NON_INDICATOR:
                yield dim_id, dim_config  # type: ignore[misc]

    @property
    def country_dimension(self) -> str | None:
        for dim_id, dim_config in self.non_indicator_dimensions:
            if dim_config.subtype is SpecialNonIndicatorDimensions.REGION:
                return dim_id
        return None

    @property
    def frequency_dimension(self) -> str | None:
        for dim_id, dim_config in self.non_indicator_dimensions:
            if dim_config.subtype is SpecialNonIndicatorDimensions.FREQUENCY:
                return dim_id
        return None

    @property
    def indicator_dimensions(self) -> list[str]:
        indicators = []
        for dim_id, dim_config in self.dimensions.items():
            if dim_config.type is DimensionType.INDICATOR:
                indicators.append(dim_id)
        return indicators

    @property
    def time_period_dimension(self) -> tuple[str, TimePeriodDimensionConfig]:
        for dim_id, dim_config in self.dimensions.items():
            if dim_config.type is DimensionType.TIME_PERIOD:
                return dim_id, dim_config  # type: ignore[return-value]
        raise ValueError("Time period dimension not found in dataset configuration")

    @property
    def time_period_dimension_id(self) -> str:
        return self.time_period_dimension[0]

    @property
    def dimension_all_values(self) -> dict[str, VirtualDimensionValue]:
        return {
            dim_id: dim_config.all_values
            for dim_id, dim_config in self.dimensions.items()
            if dim_config.all_values is not None
        }

    # ~~~~~~~~~~~~~~~~~~~~~~~ Validators ~~~~~~~~~~~~~~~~~~~~~~~

    @model_validator(mode='after')
    def no_duplicates_in_special_dimension_processors(self) -> Self:
        processor_ids = set()

        for _, sd in self.special_dimensions:
            if sd.processor_id in processor_ids:
                msg = f"Duplicate processor_id={sd.processor_id!r} found in special_dimensions"
                raise ValueError(msg)
            processor_ids.add(sd.processor_id)
        return self

    @model_validator(mode='after')
    def at_least_one_indicator_dimension(self) -> Self:
        if any(conf.type is DimensionType.INDICATOR for conf in self.dimensions.values()):
            return self
        raise ValueError("At least one indicator dimension must be defined in the dataset")

    @model_validator(mode='after')
    def exactly_one_time_period_dimension(self) -> Self:
        time_period_dims = [
            dim_id
            for dim_id, dim_config in self.dimensions.items()
            if dim_config.type is DimensionType.TIME_PERIOD
        ]
        if len(time_period_dims) != 1:
            _log.info(f"Found time period dimensions: {time_period_dims}")
            raise ValueError("Exactly one time period dimension must be defined in the dataset")
        return self

    @model_validator(mode='after')
    def one_or_none_country_dimension(self) -> Self:
        country_dims = [
            dim_id
            for dim_id, dim_config in self.non_indicator_dimensions
            if dim_config.subtype is SpecialNonIndicatorDimensions.REGION
        ]
        if len(country_dims) > 1:
            _log.info(f"Found country dimensions: {country_dims}")
            raise ValueError("At most one country dimension can be defined in the dataset")
        return self

    @model_validator(mode='after')
    def exactly_one_frequency_dimension(self) -> Self:
        frequency_dims = [
            dim_id
            for dim_id, dim_config in self.non_indicator_dimensions
            if dim_config.subtype is SpecialNonIndicatorDimensions.FREQUENCY
        ]
        if len(frequency_dims) != 1:
            _log.info(f"Found frequency dimensions: {frequency_dims}")
            raise ValueError("Exactly one frequency dimension must be defined in the dataset")
        return self

import logging
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from typing import Annotated, Literal, NamedTuple, Self

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, StrictStr, alias_generators, model_validator

from common.config.utils import replace_env
from common.utils import crc32_hash, crc32_hash_incremental

from .enums import DimensionType, SpecialNonIndicatorDimensions
from .query import Query

_log = logging.getLogger(__name__)


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)


class ConfigHashes(NamedTuple):
    indicator_hash: str
    non_indicator_hash: str
    special_hash: str


class VirtualDimensionValue(BaseModel):
    id: str
    name: str
    description: str | None


class VirtualDimensionConfig(BaseModel):
    name: str = Field(description="The name of the virtual dimension")
    description: str | None = Field(description="The description of the virtual dimension")
    value: VirtualDimensionValue = Field(description="The value of the virtual dimension")

    @property
    def indexing_hash(self) -> str:
        """A calculated hash based on the fields used for indexing."""
        data = (self.name, self.value.id, self.value.name)
        return str(crc32_hash(str(data)))


class DatasetCitation(BaseModel):
    provider: StrictStr | None = Field(default=None)
    last_updated: StrictStr | None = Field(default=None)
    url: StrictStr | None = Field(default=None)
    description: StrictStr | None = Field(default=None)

    def get_url(self) -> str | None:
        if self.url:
            return replace_env(self.url)
        return None


class IndexerIndicatorAnnotationConfig(BaseModel):
    description: str = Field(description="annotation name to get indicator description", default="")


class IndexerIndicatorConfig(BaseModel):
    unpack: bool = Field(default=False)
    use_code_list_description: bool = Field(default=False)
    super_primary: bool = Field(default=False)

    annotations: IndexerIndicatorAnnotationConfig | None = Field(default=None)


class IndexerConfig(BaseModel):
    description: str = Field(description="dataset_description", default="")

    indicator: IndexerIndicatorConfig = Field(
        description="indicator_config", default_factory=IndexerIndicatorConfig
    )

    @property
    def indexing_hash(self) -> str:
        """A calculated hash based on the fields used for indexing."""
        data = self.model_dump(
            include={
                "description",
                "indicator",
            }
        )
        return str(crc32_hash(str(data)))


class BaseDimensionConfig(BaseModel):
    dimension_type: str
    alias: str | None = Field(default=None)
    is_required: bool = Field(
        default=False,
        description=(
            "Whether this dimension is required to build a query. "
            "Used to filter out queries without these dimensions. See the detailed logic in the code"
        ),
    )
    virtual: VirtualDimensionConfig | None = Field(
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
        return DimensionType(self.dimension_type)

    @property
    def indexing_hash(self) -> str:
        """A calculated hash based on the fields used for indexing."""
        data = (
            self.dimension_type,
            self.alias,
            None if self.virtual is None else self.virtual.indexing_hash,
        )
        return str(crc32_hash(str(data)))


class IndicatorDimensionConfig(BaseDimensionConfig):
    dimension_type: Literal["INDICATOR"] = Field(default="INDICATOR")


class SpecialDimensionConfig(BaseDimensionConfig):
    dimension_type: Literal["SPECIAL"] = Field(default="SPECIAL")
    processor_id: str = Field(
        description=(
            "The ID of the processor to handle this special dimension. "
            "NOTE: processors are defined in the channel configuration."
        )
    )


class NonIndicatorDimensionConfig(BaseDimensionConfig):
    dimension_type: Literal["NON_INDICATOR"] = Field(default="NON_INDICATOR")
    subtype: SpecialNonIndicatorDimensions | None = Field(default=None)
    # named_entity: str  # TODO


class TimePeriodDimensionConfig(BaseDimensionConfig):
    dimension_type: Literal["TIME_PERIOD"] = Field(default="TIME_PERIOD")


DIMENSION_CONFIG_TYPES = Annotated[
    (
        IndicatorDimensionConfig
        | SpecialDimensionConfig
        | NonIndicatorDimensionConfig
        | TimePeriodDimensionConfig
    ),
    Field(discriminator='dimension_type'),
]


class DataSetConfig(BaseModel, ABC):
    is_official: bool = Field(default=False)
    citation: DatasetCitation | None = Field(default=None)
    indexer: IndexerConfig | None = Field(default=None)
    pinned_columns: list[str] = Field(
        description="Column names and order to pin in the data in grid", default_factory=list
    )
    dimensions: dict[str, DIMENSION_CONFIG_TYPES] = Field(
        description="The configuration of the each dimension in the dataset by its ID",
        default_factory=dict,
    )

    @abstractmethod
    def get_source_id(self) -> str:
        pass

    @property
    def indexing_hashes(self) -> ConfigHashes:
        """A calculated hash based on the fields used for indexing."""

        def get_dimensions_hash(dimensions: Iterable[tuple[str, BaseDimensionConfig]]) -> int:
            # Sort dimensions by their IDs to ensure consistent ordering
            sorted_dims = sorted(dimensions, key=lambda item: item[0])
            return crc32_hash_incremental(
                (dim_id + dim.indexing_hash) for dim_id, dim in sorted_dims
            )

        indicator_data = (
            None if self.indexer is None else self.indexer.indexing_hash,
            get_dimensions_hash(
                (i, d) for i, d in self.dimensions.items() if d.type is DimensionType.INDICATOR
            ),
        )
        return ConfigHashes(
            indicator_hash=str(crc32_hash(str(indicator_data))),
            non_indicator_hash=str(get_dimensions_hash(self.non_indicator_dimensions)),
            special_hash=str(get_dimensions_hash(self.special_dimensions)),
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

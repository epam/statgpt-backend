from pydantic import BaseModel, Field
from sdmx.message import StructureMessage
from sdmx.model.common import CubeRegion, DimensionComponent
from sdmx.model.v21 import ContentConstraint, DataStructureDefinition, MemberSelection, MemberValue

from statgpt.common.data.common.sdmx_schemas import (
    Sdmx30AnnotationModel,
    Sdmx30DataComponentFilter,
    build_availability_filters,
    to_content_constraint,
    to_structure_message,
)


class ProxyKeyValueValue(BaseModel):
    value: str = Field()

    def to_sdmx1(self) -> MemberValue:
        return MemberValue(value=self.value)


class ProxyKeyValue(BaseModel):
    key_id: str = Field(alias="id")
    include: bool = Field()
    remove_prefix: bool = Field(alias="removePrefix")
    values: list[ProxyKeyValueValue] = Field(default_factory=list)

    def to_sdmx1(self, index: int) -> MemberSelection:
        return MemberSelection(
            values=[value.to_sdmx1() for value in self.values],
            values_for=DimensionComponent(id=self.key_id, order=index),
        )


class ProxyCubeRegion(BaseModel):
    include: bool = Field()
    key_values: list[ProxyKeyValue] = Field(alias="components", default_factory=list)

    def to_sdmx1(self) -> CubeRegion:
        member_selections = [
            key_value.to_sdmx1(index) for index, key_value in enumerate(self.key_values, start=1)
        ]
        return CubeRegion(
            included=self.include,
            member={ms.values_for: ms for ms in member_selections},  # type: ignore[misc]
        )


class ProxyDataConstraint(BaseModel):
    id: str = Field()
    name: str = Field()
    names: dict[str, str] = Field(default_factory=dict)
    description: str = Field()
    descriptions: dict[str, str] = Field(default_factory=dict)
    version: str = Field()
    agency_id: str = Field(alias="agencyID")

    annotations: list["ProxyAnnotation"] = Field(default_factory=list)
    cube_regions: list[ProxyCubeRegion] = Field(alias="cubeRegions")

    def to_sdmx1(self) -> ContentConstraint:
        return to_content_constraint(
            id=self.id,
            descriptions=self.descriptions,
            names=self.names,
            version=self.version,
            agency_id=self.agency_id,
            annotations=self.annotations,
            cube_regions=self.cube_regions,
        )


class ProxyAnnotation(Sdmx30AnnotationModel):
    pass


class ProxyAvailabilityData(BaseModel):
    data_constraints: list[ProxyDataConstraint] = Field(
        alias="dataConstraints", default_factory=list
    )


class ProxyDataComponentFilter(Sdmx30DataComponentFilter):
    pass


class ProxyAvailabilityRequestBody(BaseModel):
    """A request body in the JSON format for the Proxy SDMX 3.0 API."""

    filters: list[ProxyDataComponentFilter] | None = Field(default=None)
    key: str | None = Field(default=None)
    component_id: str | None = Field(default="*")
    # ~~~ Not used: ~~~

    # updated_after: str
    # references: str
    # mode: str
    # timestampTo: datetime
    # keys: list[str]
    # skipDeleted: bool
    # dimensionAtObservation: str

    @classmethod
    def get_from(
        cls,
        key: dict[str, list[str]] | None,
        params: dict[str, str] | None,
        dsd: DataStructureDefinition | None,
    ) -> "ProxyAvailabilityRequestBody":
        return cls(
            filters=build_availability_filters(ProxyDataComponentFilter, key, params),
            key=cls._build_key_segment(key=key, dsd=dsd),
        )

    @classmethod
    def _build_key_segment(
        cls,
        *,
        key: dict[str, list[str]] | None,
        dsd: DataStructureDefinition | None,
    ) -> str:
        if not key or not dsd:
            return "*"
        dim_ids = [
            dim.id
            for dim in dsd.dimensions.components
            if not getattr(dim, "is_time_dimension", False) and dim.id != "TIME_PERIOD"
        ]
        parts = []
        for dim_id in dim_ids:
            values = key.get(dim_id)
            if not values:
                parts.append("")
            else:
                parts.append("+".join(values))
        return ".".join(parts) or "*"


class ProxyAvailabilityResponseBody(BaseModel):
    """A response body in the JSON format for Proxy SDMX 3.0 API."""

    data: ProxyAvailabilityData = Field()

    def to_sdmx1(self) -> StructureMessage:
        return to_structure_message(self.data.data_constraints)


class ProxyDataflow(BaseModel):
    annotations: list[ProxyAnnotation] = Field(default_factory=list)
    # Add other fields as needed


class ProxyStructureData(BaseModel):
    """Structure data in the JSON format for Proxy SDMX 3.0 API."""

    dataflows: list[ProxyDataflow] = Field(default_factory=list)


class ProxyDataflowMessage(BaseModel):
    """A response body in the JSON format for Proxy SDMX 3.0 API."""

    data: ProxyStructureData = Field()

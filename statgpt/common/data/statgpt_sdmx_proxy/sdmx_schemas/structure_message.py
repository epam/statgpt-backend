from pydantic import BaseModel, Field
from sdmx.message import StructureMessage
from sdmx.model.common import (
    Agency,
    AgencyScheme,
    CubeRegion,
    DimensionComponent,
    InternationalString,
)
from sdmx.model.v21 import ContentConstraint, MemberSelection, MemberValue

from statgpt.common.data.base.sdmx_schemas import (
    Sdmx30AnnotationModel,
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
    key_values: list[ProxyKeyValue] = Field(alias="keyValues", default_factory=list)

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
    description: str | None = Field(default=None)
    descriptions: dict[str, str] = Field(default_factory=dict)
    version: str = Field()
    agency_id: str = Field(alias="agencyID")

    annotations: list[Sdmx30AnnotationModel] = Field(default_factory=list)
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


class ProxyAvailabilityData(BaseModel):
    data_constraints: list[ProxyDataConstraint] = Field(
        alias="dataConstraints", default_factory=list
    )


class ProxyAvailabilityResponseBody(BaseModel):
    """A response body in the JSON format for the StatGPT SDMX proxy API."""

    data: ProxyAvailabilityData = Field()

    def to_sdmx1(self) -> StructureMessage:
        return to_structure_message(self.data.data_constraints)


class ProxyDataflow(BaseModel):
    annotations: list[Sdmx30AnnotationModel] = Field(default_factory=list)
    # Add other fields as needed


class ProxyStructureData(BaseModel):
    """Structure data in the JSON format for the StatGPT SDMX proxy API."""

    dataflows: list[ProxyDataflow] = Field(default_factory=list)


class ProxyDataflowMessage(BaseModel):
    """A response body in the JSON format for the StatGPT SDMX proxy API."""

    data: ProxyStructureData = Field()


class ProxyAgency(BaseModel):
    id: str = Field()
    name: str | None = Field(default=None)

    def to_sdmx1(self) -> Agency:
        agency = Agency(id=self.id)
        if self.name:
            agency.name = InternationalString()
            agency.name.localizations = {"en": self.name}
        return agency


class ProxyAgencyScheme(BaseModel):
    id: str = Field()
    agency_id: str = Field(alias="agencyID")
    name: str | None = Field(default=None)
    version: str = Field()
    agencies: list[ProxyAgency] = Field(default_factory=list)

    def to_sdmx1(self) -> AgencyScheme:
        scheme = AgencyScheme(id=self.id, maintainer=Agency(id=self.agency_id))
        for proxy_agency in self.agencies:
            agency = proxy_agency.to_sdmx1()
            scheme.items[agency.id] = agency
        return scheme


class ProxyAgencySchemeData(BaseModel):
    agency_schemes: list[ProxyAgencyScheme] = Field(alias="agencySchemes", default_factory=list)


class ProxyAgencySchemeResponseBody(BaseModel):
    """SDMX-JSON 2.0.0 agencyscheme response from the StatGPT SDMX proxy."""

    data: ProxyAgencySchemeData = Field()

    def to_sdmx1(self) -> StructureMessage:
        msg = StructureMessage()
        for proxy_scheme in self.data.agency_schemes:
            msg.add(proxy_scheme.to_sdmx1())
        return msg

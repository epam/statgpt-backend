from pydantic import BaseModel, Field
from sdmx.message import StructureMessage
from sdmx.model.common import CubeRegion, DimensionComponent
from sdmx.model.v21 import ContentConstraint, MemberSelection, MemberValue

from statgpt.common.data.common.sdmx_schemas import (
    Sdmx30AnnotationModel,
    Sdmx30DataComponentFilter,
    build_availability_filters,
    to_content_constraint,
    to_structure_message,
)


class QhDataComponentFilter(Sdmx30DataComponentFilter):
    pass


class QhAvailabilityRequestBody(BaseModel):
    """A request body in the JSON format for the QuantHub SDMX Plus API."""

    filters: list[QhDataComponentFilter] | None = Field(default=None)

    # ~~~ Not used: ~~~
    # key: str | None = Field(default=None)
    # updated_after: str
    # references: str
    # mode: str
    # timestampTo: datetime
    # keys: list[str]
    # skipDeleted: bool
    # dimensionAtObservation: str

    @classmethod
    def get_from(
        cls, key: dict[str, list[str]] | None, params: dict[str, str] | None
    ) -> "QhAvailabilityRequestBody":
        return cls(filters=build_availability_filters(QhDataComponentFilter, key, params))


class QhAnnotation(Sdmx30AnnotationModel):
    pass


class QhSelectionValue(BaseModel):
    member_value: str = Field(alias='memberValue')

    def to_sdmx1(self) -> MemberValue:
        return MemberValue(value=self.member_value)


class QhSelectionMember(BaseModel):
    component_id: str = Field(alias='componentId')
    selection_values: list[QhSelectionValue] = Field(alias='selectionValues', default_factory=list)

    def to_sdmx1(self, index: int) -> MemberSelection:
        return MemberSelection(
            values=[sv.to_sdmx1() for sv in self.selection_values],
            values_for=DimensionComponent(id=self.component_id, order=index),
        )


class QhCubeRegion(BaseModel):
    is_included: bool = Field(alias='isIncluded')
    member_selection: list[QhSelectionMember] = Field(alias='memberSelection', default_factory=list)

    def to_sdmx1(self) -> CubeRegion:
        member_selections = [
            ms.to_sdmx1(index) for index, ms in enumerate(self.member_selection, start=1)
        ]
        return CubeRegion(
            included=self.is_included,
            member={ms.values_for: ms for ms in member_selections},  # type: ignore[misc]
        )


class QhDataConstraint(BaseModel):
    id: str = Field()
    name: str = Field()
    names: dict[str, str] = Field(default_factory=dict)
    description: str = Field()
    descriptions: dict[str, str] = Field(default_factory=dict)
    version: str = Field()
    agency_id: str = Field(alias='agencyID')

    annotations: list[QhAnnotation] = Field(default_factory=list)
    cube_regions: list[QhCubeRegion] = Field(alias='cubeRegions')

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


class QhAvailabilityData(BaseModel):
    data_constraints: list[QhDataConstraint] = Field(alias='dataConstraints', default_factory=list)


class QhAvailabilityResponseBody(BaseModel):
    """A response body in the JSON format for the QuantHub SDMX 3.0 API."""

    data: QhAvailabilityData = Field()
    # meta: QhMeta = Field()  # Implement if needed

    def to_sdmx1(self) -> StructureMessage:
        return to_structure_message(self.data.data_constraints)


class QhDataflow(BaseModel):
    annotations: list[QhAnnotation] = Field(default_factory=list)
    # Add other fields as needed


class QhStructureData(BaseModel):
    """Structure data in the JSON format for QuantHub SDMX 3.0 API."""

    dataflows: list[QhDataflow] = Field(default_factory=list)


class QhDataflowMessage(BaseModel):
    """A response body in the JSON format for QuantHub SDMX 3.0 API."""

    data: QhStructureData = Field()
    # meta: QhMeta = Field()  # Implement if needed

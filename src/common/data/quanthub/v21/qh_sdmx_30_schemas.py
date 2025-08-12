from enum import StrEnum

from pydantic import BaseModel, Field
from sdmx.message import StructureMessage
from sdmx.model.common import Agency, CubeRegion, DimensionComponent
from sdmx.model.v21 import ContentConstraint, MemberSelection, MemberValue


class Operator(StrEnum):
    ge = "ge"
    le = "le"
    eq = "eq"


class QhDataComponentFilter(BaseModel):
    component_code: str = Field(alias='componentCode')
    operator: Operator = Field()
    value: str = Field()


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
        filters = []

        if key:
            for dim, values in key.items():
                filters.append(
                    QhDataComponentFilter(
                        componentCode=dim, operator=Operator.eq, value=",".join(values)
                    )
                )

        if params:
            if start := params.get("startPeriod"):
                start = f"{start}A" if len(start) == 4 else start  # Append 'A' for annual periods
                filters.append(
                    QhDataComponentFilter(
                        componentCode="TIME_PERIOD", operator=Operator.ge, value=start
                    )
                )
            if end := params.get("endPeriod"):
                end = f"{end}A" if len(end) == 4 else end  # Append 'A' for annual periods
                filters.append(
                    QhDataComponentFilter(
                        componentCode="TIME_PERIOD", operator=Operator.le, value=end
                    )
                )

        return cls(filters=filters)


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
    cube_regions: list[QhCubeRegion] = Field(alias='cubeRegions')

    def to_sdmx1(self) -> ContentConstraint:
        return ContentConstraint(
            id=self.id,
            description=self.descriptions,
            name=self.names,
            version=self.version,
            maintainer=Agency(id=self.agency_id),
            data_content_region=[cr.to_sdmx1() for cr in self.cube_regions],
        )


class QhAvailabilityData(BaseModel):
    data_constraints: list[QhDataConstraint] = Field(alias='dataConstraints', default_factory=list)


class QhAvailabilityResponseBody(BaseModel):
    """A response body in the JSON format for the QuantHub SDMX 3.0 API."""

    data: QhAvailabilityData = Field()
    # meta: QhMeta = Field()  # Implement if needed

    def to_sdmx1(self) -> StructureMessage:
        message = StructureMessage()
        for data_constraint in self.data.data_constraints:
            content_constraint = data_constraint.to_sdmx1()
            message.constraint[content_constraint.id] = content_constraint

        return message


class QhAnnotation(BaseModel):

    # Not sure if `id` is allowed to be None by the SDMX 3.0 standard, but some providers may return it as None.
    id: str | None = Field()

    title: str | None = Field()
    type: str | None = Field()
    value: str | None = Field()
    text: str | None = Field()


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

from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, Field
from sdmx.message import StructureMessage
from sdmx.model.common import Agency, CubeRegion, DimensionComponent
from sdmx.model.v21 import Annotation, ContentConstraint, MemberSelection, MemberValue


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


class QhAnnotation(BaseModel):

    id: str | None = Field(default=None)
    title: str | None = Field(default=None)
    type: str | None = Field(default=None)
    value: str | None = Field(default=None)
    text: str | None = Field(default=None)

    def to_sdmx1(self) -> Annotation:
        return Annotation(
            id=self.id,
            title=self.title,
            type=self.type,
            text=self.text,
            # The `value` field was added by SDMX 3.0.0, so it's not included here.
        )


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
            key_value.to_sdmx1(index)
            for index, key_value in enumerate(self.key_values, start=1)
        ]
        return CubeRegion(
            included=self.include,
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
        return ContentConstraint(
            id=self.id,
            description=self.descriptions,
            name=self.names,
            version=self.version,
            maintainer=Agency(id=self.agency_id),
            annotations=[a.to_sdmx1() for a in self.annotations],
            data_content_region=[cr.to_sdmx1() for cr in self.cube_regions],
        )


class ProxyDataConstraint(BaseModel):
    id: str = Field()
    name: str = Field()
    names: dict[str, str] = Field(default_factory=dict)
    description: str = Field()
    descriptions: dict[str, str] = Field(default_factory=dict)
    version: str = Field()
    agency_id: str = Field(alias='agencyID')

    annotations: list[QhAnnotation] = Field(default_factory=list)
    cube_regions: list[ProxyCubeRegion] = Field(alias='cubeRegions')

    def to_sdmx1(self) -> ContentConstraint:
        return ContentConstraint(
            id=self.id,
            description=self.descriptions,
            name=self.names,
            version=self.version,
            maintainer=Agency(id=self.agency_id),
            annotations=[a.to_sdmx1() for a in self.annotations],
            data_content_region=[cr.to_sdmx1() for cr in self.cube_regions],
        )


class QhAvailabilityData(BaseModel):
    data_constraints: list[QhDataConstraint] = Field(alias='dataConstraints', default_factory=list)


class QhAvailabilityResponseBody(BaseModel):
    """A response body in the JSON format for the QuantHub SDMX 3.0 API."""

    data: QhAvailabilityData = Field()
    # meta: QhMeta = Field()  # Implement if needed

    def to_sdmx1(self) -> StructureMessage:
        return _to_structure_message(self.data.data_constraints)


class ProxyAvailabilityData(BaseModel):
    data_constraints: list[ProxyDataConstraint] = Field(alias='dataConstraints', default_factory=list)


class ProxyAvailabilityResponseBody(BaseModel):
    """A response body in the JSON format for Proxy SDMX 3.0 API."""

    data: ProxyAvailabilityData = Field()

    def to_sdmx1(self) -> StructureMessage:
        return _to_structure_message(self.data.data_constraints)


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


class _SdmxConstraint(Protocol):
    def to_sdmx1(self) -> ContentConstraint:
        ...


def _to_structure_message(data_constraints: Iterable[_SdmxConstraint]) -> StructureMessage:
    message = StructureMessage()
    for data_constraint in data_constraints:
        content_constraint = data_constraint.to_sdmx1()
        message.constraint[content_constraint.id] = content_constraint
    return message

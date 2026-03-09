from collections.abc import Iterable
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field
from sdmx.message import StructureMessage
from sdmx.model.common import Agency
from sdmx.model.v21 import Annotation, ContentConstraint


class Operator(StrEnum):
    ge = "ge"
    le = "le"
    eq = "eq"


class Sdmx30AnnotationModel(BaseModel):
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


class Sdmx30DataComponentFilter(BaseModel):
    component_code: str = Field(alias="componentCode")
    operator: Operator = Field()
    value: str = Field()


def build_availability_filters(
    key: dict[str, list[str]] | None,
    params: dict[str, str] | None,
) -> list[Sdmx30DataComponentFilter]:
    filters: list[Sdmx30DataComponentFilter] = []

    if key:
        for dim, values in key.items():
            filters.append(
                Sdmx30DataComponentFilter(
                    componentCode=dim,
                    operator=Operator.eq,
                    value=",".join(values),
                )
            )

    if params:
        if start := params.get("startPeriod"):
            start = f"{start}A" if len(start) == 4 else start
            filters.append(
                Sdmx30DataComponentFilter(
                    componentCode="TIME_PERIOD",
                    operator=Operator.ge,
                    value=start,
                )
            )
        if end := params.get("endPeriod"):
            end = f"{end}A" if len(end) == 4 else end
            filters.append(
                Sdmx30DataComponentFilter(
                    componentCode="TIME_PERIOD",
                    operator=Operator.le,
                    value=end,
                )
            )

    return filters


class PostAvailabilityRequestBody(BaseModel):
    """A POST request body for SDMX 3.0 availability endpoints."""

    filters: list[Sdmx30DataComponentFilter] | None = Field(default=None)

    @classmethod
    def get_from(
        cls, key: dict[str, list[str]] | None, params: dict[str, str] | None
    ) -> "PostAvailabilityRequestBody":
        return cls(filters=build_availability_filters(key, params))


class _SdmxConstraint(Protocol):
    def to_sdmx1(self) -> ContentConstraint: ...


def to_structure_message(data_constraints: Iterable[_SdmxConstraint]) -> StructureMessage:
    message = StructureMessage()
    for data_constraint in data_constraints:
        content_constraint = data_constraint.to_sdmx1()
        message.constraint[content_constraint.id] = content_constraint
    return message


def to_content_constraint(
    *,
    id: str,
    descriptions: dict[str, str],
    names: dict[str, str],
    version: str,
    agency_id: str,
    annotations: Iterable[Any],
    cube_regions: Iterable[Any],
) -> ContentConstraint:
    return ContentConstraint(
        id=id,
        description=descriptions,
        name=names,
        version=version,
        maintainer=Agency(id=agency_id),
        annotations=[a.to_sdmx1() for a in annotations],
        data_content_region=[cr.to_sdmx1() for cr in cube_regions],
    )

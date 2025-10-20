from typing import Any, Literal

from pydantic import BaseModel, Field

# ValueArray = list[int | None] | list[str | None] | list[bool | None] | list['ValueArray']
ValueArray = list[Any]


class QhDataSet(BaseModel):
    structure: int = Field(
        default=0,
        description=(
            "Structure contains the index of the structure in the structures array of the data object,"
            " which describe the data in this dataSet. If omitted, then the data in this dataSet are"
            " assumed to be described by the first structure"
        ),
    )
    attributes: list[ValueArray | Literal[0] | None] = Field(
        default_factory=list,
        description="This is a list of attribute values or indices of the corresponding values.",
    )
    # Add other fields as needed


class Component(BaseModel):
    """A component represents a `dimension`, a `measure` or an `attribute` used in the message.
    It contains basic information about the component (such as its `name` and `id`)
    as well as the list of `values` used in the message for this particular component.
    """

    id: str = Field()
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    is_mandatory: bool = Field(alias='isMandatory', default=False)
    values: list[Any] = Field(default_factory=list)
    # Add other fields as needed


class Attribute(Component):
    """Describes a single attribute used in the message."""


class Attributes(BaseModel):
    """Describes the attributes used in the message."""

    data_set: list[Attribute] = Field(
        alias='dataSet', default_factory=list, description="DataSet-level attributes."
    )
    dimension_group: list[Attribute] = Field(
        alias='dimensionGroup', default_factory=list, description="DimensionGroup-level attributes."
    )
    series: list[Attribute] = Field(default_factory=list, description="Series-level attributes.")
    observation: list[Attribute] = Field(
        default_factory=list, description="Observation-level attributes."
    )


class QhStructure(BaseModel):
    """Structure information of the SDMX 3.0 Data Message."""

    links: list = Field(default_factory=list)
    data_sets: list[int] | None = Field(
        alias='dataSets',
        default=None,
        description=(
            "It contains the indexes of the dataSet objects in the dataSets array of the data object,"
            " which contain the data for this structure. If omitted, then all data included in the"
            " message are assumed to be described by this structure."
        ),
    )
    attributes: Attributes | None = Field(default=None)
    # Add other fields as needed


class QhData(BaseModel):
    """Primary data of the SDMX 3.0 Data Message."""

    structures: list[QhStructure] = Field(default_factory=list)
    data_sets: list[QhDataSet] = Field(
        alias='dataSets',
        default_factory=list,
        description="In typical cases, this field will contain only one DataSet.",
    )


class QhDataMessage(BaseModel):
    """A response body in the JSON format for QuantHub SDMX 3.0 Data API."""

    # meta: QhMeta = Field()  # Implement if needed
    data: QhData | None = Field(default=None)
    # errors: list = Field(default_factory=list)  # Implement if needed

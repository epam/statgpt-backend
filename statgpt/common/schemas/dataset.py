import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from .auditable import Auditable
from .base import DbDefaultBase
from .data_source import DataSource


class Status(BaseModel):
    status: Literal['online', 'offline', 'invalid_config'] = Field(
        description="The status of the dataset"
    )
    details: str = Field(description="The details of the dataset status", default="")


class DataSetBase(BaseModel):
    id_: uuid.UUID = Field(default_factory=uuid.uuid4, description="The uuid of the dataset")
    title: str
    data_source_id: int
    details: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dataset config as a JSON object. "
            "The particular schema depends on the specific implementation of the data source."
        ),
    )


class DataSetDescriptor(BaseModel):
    data_source_id: int = Field(description="The ID of the data source")
    id_in_source: str = Field(description="The ID of the dataset in the data source")
    title: str
    description: str = Field(default="")
    details: dict[str, Any]


class DataSet(DataSetBase, DbDefaultBase, Auditable):
    description: str = Field(default="")
    data_source: DataSource | None

    status: Status

    @computed_field(  # type: ignore[prop-decorator]
        deprecated=True,
        description='This field is added for backward compatibility with the admin interface.'
        ' In new frontend code, use `status` instead.',
    )
    @property
    def preprocessing_status(self) -> str:
        return self.status.status

    def get_entity_id(self) -> str:
        return str(self.id_)

    def get_entity_name(self) -> str:
        return self.title

    def get_state_after(self) -> dict:
        return self.model_dump(
            mode='json',
            exclude={"created_at", "updated_at", "description", "data_source"},
        )

    def get_item_id(self) -> int:
        return self.id


class DataSetUpdateRequest(BaseModel):
    title: str | None = Field(default=None)
    data_source_id: int | None = Field(default=None)
    details: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Dataset config as a JSON object. "
            "The particular schema depends on the specific implementation of the data source."
        ),
    )

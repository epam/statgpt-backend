import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from .base import DbDefaultBase
from .data_source import DataSource


class Status(BaseModel):
    status: Literal['online', 'offline'] = Field(description="The status of the dataset")
    details: str = Field(description="The details of the dataset status", default="")


class DataSetBase(BaseModel):
    id_: uuid.UUID
    title: str
    data_source_id: int
    details: dict[str, Any] = Field(default_factory=dict, description="Details as a JSON object")


class DataSetDescriptor(BaseModel):
    title: str
    description: str = Field(default="")
    data_source_id: int
    details: dict[str, Any]


class DataSet(DataSetBase, DbDefaultBase):
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


class DataSetUpdate(BaseModel):
    title: str | None = Field(default=None)
    data_source_id: int | None = Field(default=None)
    details: dict[str, Any] | None = Field(default=None, description="Details as a JSON object")

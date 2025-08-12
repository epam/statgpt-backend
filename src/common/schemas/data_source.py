from typing import Any

from pydantic import BaseModel, Field

from .base import DbDefaultBase


class DataSourceType(DbDefaultBase):
    name: str
    description: str


class DataSourceBase(BaseModel):
    title: str
    description: str = ""
    type_id: int
    details: dict[str, Any] = Field(default_factory=dict, description="Details as a JSON object")


class DataSourceUpdate(BaseModel):
    title: str | None = Field(default=None)
    description: str | None = Field(default=None)
    details: dict[str, Any] | None = Field(default=None, description="Details as a JSON object")


class DataSource(DbDefaultBase, DataSourceBase):
    type: DataSourceType

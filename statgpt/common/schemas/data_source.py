from typing import Any

from pydantic import BaseModel, Field

from .auditable import Auditable
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


class DataSource(DbDefaultBase, DataSourceBase, Auditable):
    type: DataSourceType

    def get_entity_id(self) -> str | None:
        return self.title

    def get_entity_name(self) -> str | None:
        return self.description

    def get_state_after(self) -> dict | None:
        return self.model_dump(exclude_none=True)

    def get_item_id(self) -> int | None:
        return self.id

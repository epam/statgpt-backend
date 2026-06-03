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

    def get_entity_id(self) -> str:
        return self.title

    def get_entity_name(self) -> str:
        return self.description

    def get_state_after(self) -> dict:
        return self.model_dump(mode='json', exclude={"created_at", "updated_at"})

    def get_item_id(self) -> int:
        return self.id


class Provider(BaseModel):
    id: str = Field(description="Provider (agency) id, e.g. 'IMF.RES'")
    name: str = Field(
        description="Display name of the provider; falls back to the id when unknown."
    )

import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, alias_generators


class DbDefaultBase(BaseModel):
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


ItemT = TypeVar("ItemT")


class ListResponse(BaseModel, Generic[ItemT]):
    data: list[ItemT]

    limit: int
    offset: int

    count: int
    total: int


class BaseYamlModel(BaseModel):
    model_config = ConfigDict(alias_generator=alias_generators.to_camel, populate_by_name=True)


class SystemUserPrompt(BaseYamlModel):
    """prompt consisting of 2 messages: system and user"""

    system_message: str
    user_message: str

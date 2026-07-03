import datetime
import typing as t
from typing import Generic, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, alias_generators

from statgpt.common.utils.files import read_yaml


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
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel, populate_by_name=True, extra="ignore"
    )


class DefaltPromptsBase(BaseModel):
    model_config = ConfigDict(
        frozen=True,  # prevent any modifications
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
    )

    @classmethod
    def from_yaml(cls, fp) -> t.Self:
        prompts_raw = read_yaml(fp)
        return cls.model_validate(prompts_raw)


class SystemUserPrompt(BaseYamlModel):
    """prompt consisting of 2 messages: system and user"""

    system_message: str
    user_message: str

    def get_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_message),
                ("human", self.user_message),
            ]
        )

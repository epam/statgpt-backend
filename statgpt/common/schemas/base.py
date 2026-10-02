import datetime
import typing as t
from typing import Generic, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, alias_generators, field_validator

from statgpt.common.config.utils import replace_env
from statgpt.common.utils.files import read_yaml


def bool_from_str(value: str) -> bool:
    """
    Converts a string to a boolean value.
    If the string is an environment variable reference, it will be replaced with its value before conversion.
    """
    return TypeAdapter(bool).validate_python(replace_env(value))


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


class ToggleableConfig(BaseYamlModel):
    """A config block with an on/off flag that may reference an environment variable."""

    enabled_str: str = Field(
        description=(
            "Whether the feature is enabled."
            " The value can be a reference to an environment variable."
        )
    )

    @field_validator('enabled_str', mode='after')
    @classmethod
    def validate_enabled(cls, enabled: str) -> str:
        """Validate the `enabled` field to ensure it can return a boolean value."""
        try:
            bool_from_str(enabled)
        except Exception as e:
            raise ValueError(f"Invalid value for enabled_str: {enabled}. Error: {e}")
        return enabled

    @property
    def enabled(self) -> bool:
        return bool_from_str(self.enabled_str)


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

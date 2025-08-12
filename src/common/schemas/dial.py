"""A copy of the `aidial_sdk` schemas, but we inherit them from pydantic v2 base model"""

from typing import Any, Literal

from aidial_sdk.chat_completion import Role, Status
from pydantic import BaseModel, ConfigDict, Field, StrictStr


class ExtraAllowModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='allow')


class Attachment(ExtraAllowModel):
    type: StrictStr | None = Field(default="text/markdown")
    title: StrictStr | None = Field(default=None)
    data: StrictStr | None = Field(default=None)
    url: StrictStr | None = Field(default=None)
    reference_type: StrictStr | None = Field(default=None)
    reference_url: StrictStr | None = Field(default=None)


class Stage(ExtraAllowModel):
    name: StrictStr
    status: Status
    content: StrictStr | None = Field(default=None)
    attachments: list[Attachment] | None = Field(default=None)


class CustomContent(ExtraAllowModel):
    stages: list[Stage] | None = Field(default=None)
    attachments: list[Attachment] | None = Field(default=None)
    state: Any | None = Field(default=None)
    form_value: Any | None = None
    form_schema: Any | None = None


class CacheBreakpoint(ExtraAllowModel):
    expire_at: StrictStr | None = None


class MessageCustomFields(ExtraAllowModel):
    cache_breakpoint: CacheBreakpoint | None = None


class FunctionCall(ExtraAllowModel):
    name: str
    arguments: str


class ToolCall(ExtraAllowModel):
    # OpenAI API doesn't strictly specify existence of the index field
    index: int | None
    id: StrictStr
    type: Literal["function"]
    function: FunctionCall


class Message(ExtraAllowModel):
    role: Role
    content: StrictStr | None = Field(default=None)
    custom_content: CustomContent | None = Field(default=None)
    custom_fields: MessageCustomFields | None = None
    name: StrictStr | None = Field(default=None)
    tool_calls: list[ToolCall] | None = Field(default=None)
    tool_call_id: StrictStr | None = Field(default=None)
    function_call: FunctionCall | None = Field(default=None)


class Pricing(BaseModel):
    unit: str
    prompt: float
    completion: float

from typing import Annotated, ClassVar

from aidial_sdk.chat_completion import Request
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages import ToolCall as LangChainToolCall
from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ConfigDict, Field, computed_field

from statgpt.app.settings.dial_app import dial_app_settings

StatGPTMessage = Annotated[ToolMessage | AIMessage | SystemMessage, Field(discriminator="type")]


class State(BaseModel):
    CMD_PREFIX: ClassVar[str] = 'cmd_'
    SHOW_DEBUG_STAGES: ClassVar[str] = "show_debug_stages"

    show_debug_stages: bool = Field(default=dial_app_settings.dial_show_debug_stages)
    cmd_out_of_scope_only: bool = Field(default=dial_app_settings.cmd_out_of_scope_only)
    cmd_rag_prefilter_only: bool = Field(default=dial_app_settings.cmd_rag_prefilter_only)
    cmd_skip_data_query_summarization: bool = Field(
        default=dial_app_settings.cmd_skip_data_query_summarization
    )
    cmd_skip_tools_execution: bool = Field(default=dial_app_settings.cmd_skip_tools_execution)
    out_of_scope: bool | None = Field(default=None)
    out_of_scope_reasoning: str | None = Field(default=None)

    error: str | None = Field(default=None)
    direct_tool_calls: list[LangChainToolCall] = Field(default_factory=list)
    tool_messages: list[StatGPTMessage] = Field(default_factory=list)

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    @classmethod
    def init_state(cls, request: Request) -> "State":
        if len(request.messages) < 2:
            return cls()

        last_response = request.messages[-2]
        if custom_content := last_response.custom_content:
            return cls.model_validate(custom_content.state or {})
        else:
            return cls()

    @classmethod
    def get_intercaptable_commands(cls) -> list[tuple[str, str]]:
        commands = []
        for field_name, _ in State.model_fields.items():
            commands.append((field_name.replace(cls.CMD_PREFIX, ""), field_name))
        return commands

    @property
    @computed_field
    def any_tool_response_failed(self):
        return any(
            msg.status == "error" for msg in self.tool_messages if isinstance(msg, ToolMessage)
        )

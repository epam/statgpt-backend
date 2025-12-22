from __future__ import annotations

import re
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
    def init_from_request(cls, request: Request) -> "State":
        if len(request.messages) < 2:
            return cls()

        last_response = request.messages[-2]
        if custom_content := last_response.custom_content:
            return cls.model_validate(custom_content.state or {})
        else:
            return cls()

    @classmethod
    def get_intercaptable_commands(cls, include_dev_commands: bool) -> list[InterceptableCommand]:
        commands = [
            InterceptableCommand(command="show_debug_stages", state_var="show_debug_stages")
        ]
        if include_dev_commands:
            for field_name, _ in State.model_fields.items():
                if field_name.startswith(cls.CMD_PREFIX):
                    commands.append(
                        InterceptableCommand(
                            command=field_name.removeprefix(cls.CMD_PREFIX), state_var=field_name
                        )
                    )
        return commands

    @property
    @computed_field
    def any_tool_response_failed(self):
        return any(
            msg.status == "error" for msg in self.tool_messages if isinstance(msg, ToolMessage)
        )


class InterceptableCommand(BaseModel):
    command: str
    state_var: str

    @property
    def re_pattern(self) -> str:
        return rf'!{self.command}(\s+)'

    def process_query(self, query: str, state: State | None) -> str:
        """
        To correctly parse command, it must have a space afterwards, refer to the regex pattern used
        for matching and defined in the `re_pattern` property.
        """
        match = re.search(self.re_pattern, query)
        if not match:
            return query

        # at least one command instance found

        if state is not None:
            setattr(state, self.state_var, True)

        # remove all command instances from the query
        query_edited = re.sub(self.re_pattern, '', query)
        query_edited = query_edited.strip()
        return query_edited

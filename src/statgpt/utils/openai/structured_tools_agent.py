import json
import typing as t
from json import JSONDecodeError
from typing import List, Union

from langchain.agents.agent import MultiActionAgentOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolAgentAction
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_pydantic_to_openai_function,
    convert_to_openai_tool,
)
from pydantic import BaseModel

_SCHEMA_TOOL_DESCRIPTION = """\
Structured output tool. Call this tool only when you are ready to provide final result.\
"""


def _parse_ai_message_to_openai_tool_action(
    message: BaseMessage, schema: t.Type[BaseModel], schema_tool_name: str
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    if not message.additional_kwargs.get("tool_calls"):
        return AgentFinish(return_values={"output": message.content}, log=str(message.content))

    actions: List = []
    for tool_call in message.additional_kwargs["tool_calls"]:
        function = tool_call["function"]
        function_name = function["name"]
        try:
            _tool_input = json.loads(function["arguments"] or "{}")
        except JSONDecodeError:
            raise OutputParserException(
                f"Could not parse tool input: {function} because "
                f"the `arguments` is not valid JSON."
            )

        # HACK HACK HACK:
        # The code that encodes tool input into Open AI uses a special variable
        # name called `__arg1` to handle old style tools that do not expose a
        # schema and expect a single string argument as an input.
        # We unpack the argument here if it exists.
        # Open AI does not support passing in a JSON array as an argument.
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = f"responded: {message.content}\n" if message.content else "\n"
        log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
        if function_name == schema_tool_name:
            result_output = schema(**tool_input)
            return AgentFinish(return_values={"output": result_output}, log=str(message.content))
        actions.append(
            OpenAIToolAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
                tool_call_id=tool_call["id"],
            )
        )
    return actions


TBaseModel = t.TypeVar("TBaseModel", bound=BaseModel)


class OpenAIToolsAgentStructuredOutputParser(MultiActionAgentOutputParser, t.Generic[TBaseModel]):

    structured_output_schema: t.Type[BaseModel]
    structured_output_schema_name: str

    @property
    def _type(self) -> str:
        return "openai-tools-agent-structured-output-parser"

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[List[AgentAction], AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return _parse_ai_message_to_openai_tool_action(
            message, self.structured_output_schema, self.structured_output_schema_name
        )

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")


def create_openai_schema_tool(schema: t.Type[BaseModel]) -> t.Dict[str, t.Any]:
    return {
        "type": "function",
        "function": convert_pydantic_to_openai_function(
            schema, description=_SCHEMA_TOOL_DESCRIPTION
        ),
    }


# NOTE: deprecated
def create_openai_tools_agent_with_structured_output(
    llm: BaseLanguageModel,
    tools: t.Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    schema: t.Type[BaseModel],
) -> t.Tuple[Runnable, t.Sequence[BaseTool]]:
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    openai_schema_tool = create_openai_schema_tool(schema)
    openai_schema_tool_name = openai_schema_tool["function"]["name"]

    openai_tools = [
        *[convert_to_openai_tool(tool) for tool in tools],
        openai_schema_tool,
    ]

    llm_with_tools = llm.bind(tools=openai_tools)

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(x["intermediate_steps"])
        )
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentStructuredOutputParser(
            structured_output_schema=schema,
            structured_output_schema_name=openai_schema_tool_name,
        )
    )
    return agent, tools

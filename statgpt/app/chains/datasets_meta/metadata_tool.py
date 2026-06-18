from typing import Any

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from mcp.types import ToolAnnotations
from pydantic import Field

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.tools import StatGptTool, ToolArgs
from statgpt.app.default_prompts import datasets_metadata_default_prompts
from statgpt.app.schemas import ToolArtifact, ToolMessageState
from statgpt.app.utils.formatters import DatasetsListFormatter
from statgpt.common.schemas import ChannelConfig
from statgpt.common.schemas import DatasetsMetadataTool as DatasetsMetadataToolConfig
from statgpt.common.schemas import ToolTypes
from statgpt.common.schemas.enums import AvailableDatasetsVersion
from statgpt.common.utils.models import get_chat_model

from ._utils import _create_formatter_config


class DatasetsMetadataArgs(ToolArgs):
    query: str = Field(
        description="Natural language query that can be answered with datasets metadata."
    )


class DatasetsMetadataTool(
    StatGptTool[DatasetsMetadataToolConfig], tool_type=ToolTypes.DATASETS_METADATA
):
    @classmethod
    def get_mcp_annotations(cls) -> ToolAnnotations:
        return ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    def __init__(
        self, tool_config: DatasetsMetadataToolConfig, channel_config: ChannelConfig, **kwargs
    ):
        super().__init__(tool_config, channel_config, **kwargs)
        self._dataset_formatter_config = _create_formatter_config(
            AvailableDatasetsVersion.full, channel_config.locale
        )
        if not tool_config.details.system_prompt:
            tool_config.details.system_prompt = datasets_metadata_default_prompts.system_prompt

    @classmethod
    def get_args_schema(cls, tool_config: DatasetsMetadataToolConfig) -> type[DatasetsMetadataArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return DatasetsMetadataArgs

    def get_guardrail_input(self, arguments: dict[str, Any]) -> str | None:
        return arguments.get("query")

    async def _arun(self, inputs: dict, query: str, **kwargs) -> tuple[str, ToolArtifact]:
        data_service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        target = ChainParameters.get_target(inputs)

        versioned_datasets = await data_service.list_available_datasets(auth_context)
        datasets = [ds.data for ds in versioned_datasets]

        formatter = DatasetsListFormatter(self._dataset_formatter_config, auth_context=auth_context)
        datasets_formatted = await formatter.format(
            datasets, sort_by_name=True, add_stats=True, group_by_provider=True
        )

        params = dict(
            datasets=datasets_formatted,
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._tool_config.details.system_prompt),  # type: ignore[arg-type]
                ("human", "{query}"),
            ]
        ).partial(**params)

        llm = get_chat_model(
            api_key=auth_context.api_key, model_config=self._tool_config.details.llm_model_config
        )

        chain = prompt_template | llm
        response = ""

        async for chunk in chain.astream(dict(query=query)):
            content = chunk.content
            response += content  # type: ignore[operator]
            if target:
                target.append_content(content)  # type: ignore[arg-type]

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))

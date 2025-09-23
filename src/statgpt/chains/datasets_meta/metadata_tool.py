from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from pydantic import Field

from common.schemas import ChannelConfig
from common.schemas import DatasetsMetadataTool as DatasetsMetadataToolConfig
from common.schemas import ToolTypes
from common.schemas.enums import AvailableDatasetsVersion
from common.utils.models import get_chat_model
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool, ToolArgs
from statgpt.default_prompts.prompts import DatasetsMetadataPrompts
from statgpt.schemas import ToolArtifact, ToolMessageState
from statgpt.utils.formatters import DatasetsListFormatter

from ._utils import _create_formatter_config


class DatasetsMetadataArgs(ToolArgs):
    query: str = Field(
        description="Natural language query that can be answered with datasets metadata."
    )


class DatasetsMetadataTool(
    StatGptTool[DatasetsMetadataToolConfig], tool_type=ToolTypes.DATASETS_METADATA
):
    def __init__(
        self, tool_config: DatasetsMetadataToolConfig, channel_config: ChannelConfig, **kwargs
    ):
        super().__init__(tool_config, channel_config, **kwargs)
        self._dataset_formatter_config = _create_formatter_config(
            AvailableDatasetsVersion.full, channel_config.locale
        )
        if not tool_config.details.system_prompt:
            tool_config.details.system_prompt = DatasetsMetadataPrompts.METADATA_SYSTEM_PROMPT

    @classmethod
    def get_args_schema(cls, tool_config: DatasetsMetadataToolConfig) -> type[DatasetsMetadataArgs]:
        """Return the schema for the arguments that this tool accepts."""
        return DatasetsMetadataArgs

    async def _arun(self, inputs: dict, query: str, **kwargs) -> tuple[str, ToolArtifact]:
        data_service = ChainParameters.get_data_service(inputs)
        auth_context = ChainParameters.get_auth_context(inputs)
        datasets = await data_service.list_available_datasets(auth_context)
        target = ChainParameters.get_target(inputs)

        datasets_formatted = await DatasetsListFormatter(
            self._dataset_formatter_config,
            auth_context=auth_context,
        ).format(datasets, sort_by_name=True, add_stats=True, group_by_provider=True)

        params = dict(
            datasets=datasets_formatted,
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self._tool_config.details.system_prompt),
                ("human", "{query}"),
            ]
        ).partial(**params)

        llm = get_chat_model(
            api_key=auth_context.api_key,
            model_config=self._tool_config.details.llm_model_config,
        )

        chain = prompt_template | llm
        response = ""

        async for chunk in chain.astream(
            dict(
                query=query,
            )
        ):
            content = chunk.content
            response += content
            target.append_content(content)

        return response, ToolArtifact(state=ToolMessageState(type=self.tool_type))

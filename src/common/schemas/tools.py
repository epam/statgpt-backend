from pydantic import Field

from .base import BaseYamlModel
from .data_query_tool import DataQueryDetails
from .enums import ToolTypes
from .tool_details import (
    AvailablePublicationsDetails,
    BaseToolDetails,
    FileRagDetails,
    PlainContentDetails,
    TermDefinitionsDetails,
    WebSearchAgentDetails,
    WebSearchDetails,
)


class BaseToolConfig(BaseYamlModel):
    type: ToolTypes

    # The restrictions below are defined by OpenAI
    name: str = Field(
        description="The name of the tool. Must be unique within a channel.",
        pattern=r'^[a-zA-Z0-9_\.-]+$',
    )
    description: str = Field(description="The description of the tool.", max_length=1024)

    details: BaseToolDetails = Field(
        default_factory=BaseToolDetails, description="Details as a JSON object"
    )

    @property
    def out_of_scope_description(self) -> str:
        return self.description


class AvailableDatasetsTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.AVAILABLE_DATASETS


class DataQueryTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.DATA_QUERY
    details: DataQueryDetails = Field(default_factory=DataQueryDetails)


class FileRagTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.FILE_RAG
    details: FileRagDetails = Field(default_factory=FileRagDetails)  # type: ignore


class WebSearchTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.WEB_SEARCH
    details: WebSearchDetails = Field(default_factory=WebSearchDetails)  # type: ignore

    @property
    def out_of_scope_description(self) -> str:
        if domains_config := self.details.domains:
            return f"{self.description}\n\n{domains_config.field_name}: {domains_config.allowed_values}"
        return self.description


class WebSearchAgentTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.WEB_SEARCH_AGENT
    details: WebSearchAgentDetails = Field(default_factory=WebSearchAgentDetails)  # type: ignore


class AvailablePublicationsTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.AVAILABLE_PUBLICATIONS
    details: AvailablePublicationsDetails = Field(default_factory=AvailablePublicationsDetails)  # type: ignore


class PlainContentTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.PLAIN_CONTENT
    details: PlainContentDetails = Field(default_factory=PlainContentDetails)


class AvailableTermsTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.AVAILABLE_TERMS
    details: BaseToolDetails = Field(default_factory=BaseToolDetails)


class TermDefinitionsTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.TERM_DEFINITIONS
    details: TermDefinitionsDetails = Field(default_factory=TermDefinitionsDetails)

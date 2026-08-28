from typing import Literal

from pydantic import Field

from .base import BaseYamlModel
from .data_query_tool import DataQueryDetails
from .dataset_structure_tool import DatasetStructureToolDetails
from .discovery_datasets_tool import DiscoveryDatasetsDetails
from .enums import ToolTypes
from .tool_details import (
    AvailableDatasetsDetails,
    AvailablePublicationsDetails,
    AvailableTermsDetails,
    BaseToolDetails,
    DatasetsMetadataDetails,
    DeepResearchDetails,
    FileRagDetails,
    PlainContentDetails,
    SdmxQueryAppDetails,
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
    description: str = Field(description="The description of the tool.", max_length=4096)
    enabled: bool = Field(default=True, description="Whether the tool is enabled or not.")

    mcp_only: bool = Field(
        default=False,
        description=(
            "If True, the tool is surfaced only via the MCP server and excluded from the"
            " Supreme Agent (the LLM cannot call it). Used for UI-initiated tool calls."
        ),
    )

    mcp_visibility: list[Literal["model", "app"]] | None = Field(
        default=None,
        description=(
            'MCP-App visibility per the MCP Apps spec (`_meta.ui.visibility`): a list'
            ' containing "model" and/or "app". Omit for the spec default ["model", "app"];'
            ' use ["app"] to hide from the model, ["model"] to hide from the app.'
            " Independent of `mcp_only` (agent-exclusion only)."
        ),
    )

    mcp_app_resource_uri: str | None = Field(
        default=None,
        description=(
            "MCP-App UI resource bound to this tool (`_meta.ui.resourceUri`). When set, the host"
            " can preload/render the widget at this `ui://` URI when the tool is called. Must"
            " reference a resource declared in the channel's `mcp.resources`."
        ),
    )

    mcp_name: str | None = Field(
        default=None,
        pattern=r'^[a-zA-Z0-9_\.-]+$',
        description="Override for the tool name exposed via MCP. Defaults to `name` if not set.",
    )
    mcp_description: str | None = Field(
        default=None,
        max_length=4096,
        description=(
            "Override for the tool description exposed via MCP."
            " Defaults to `description` if not set."
        ),
    )

    details: BaseToolDetails = Field(
        default_factory=BaseToolDetails, description="Details as a JSON object"
    )

    @property
    def out_of_scope_description(self) -> str:
        return self.description

    @property
    def effective_mcp_name(self) -> str:
        return self.mcp_name if self.mcp_name is not None else self.name

    @property
    def effective_mcp_description(self) -> str:
        return self.mcp_description if self.mcp_description is not None else self.description


class AvailableDatasetsTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.AVAILABLE_DATASETS
    details: AvailableDatasetsDetails = Field(default_factory=AvailableDatasetsDetails)


class DatasetsMetadataTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.DATASETS_METADATA
    details: DatasetsMetadataDetails = Field(default_factory=DatasetsMetadataDetails)


class DatasetStructureTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.DATASET_STRUCTURE
    details: DatasetStructureToolDetails = Field(default_factory=DatasetStructureToolDetails)


class DataQueryTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.DATA_QUERY
    details: DataQueryDetails = Field(default_factory=DataQueryDetails)


class DiscoveryDatasetsTool(BaseToolConfig):
    """Discovery datasets: the publish target, and the chat-time lookup over what was published.

    Not listed in `ChannelConfig.tool_fields` - the lookup is run by the data query tool, not the
    Supreme Agent - so `enabled` gates only that lookup, never publishing or indexing.
    """

    type: ToolTypes = ToolTypes.DISCOVERY_DATASETS
    details: DiscoveryDatasetsDetails


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


class DeepResearchTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.DEEP_RESEARCH
    details: DeepResearchDetails = Field(default_factory=DeepResearchDetails)  # type: ignore


class AvailablePublicationsTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.AVAILABLE_PUBLICATIONS
    details: AvailablePublicationsDetails = Field(default_factory=AvailablePublicationsDetails)  # type: ignore


class PlainContentTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.PLAIN_CONTENT
    details: PlainContentDetails = Field(default_factory=PlainContentDetails)


def _app_only_visibility() -> list[Literal["model", "app"]]:
    return ["app"]


class SdmxQueryAppTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.SDMX_QUERY_APP
    # MCP-only by design: the request is built and invoked by the MCP-App component,
    # never by the Supreme Agent. Pinned to True so a YAML config cannot opt out.
    mcp_only: Literal[True] = True
    # App-only by default so the model never sees this passthrough tool; still overridable.
    mcp_visibility: list[Literal["model", "app"]] | None = Field(
        default_factory=_app_only_visibility
    )
    details: SdmxQueryAppDetails


class DatasetsMetadataAppTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.DATASETS_METADATA_APP
    # MCP-only by design: consumed by the MCP-App (UI widget), never by the Supreme Agent.
    # Pinned to True so a YAML config cannot opt out.
    mcp_only: Literal[True] = True
    # App-only by default so the model never sees this metadata tool; still overridable.
    mcp_visibility: list[Literal["model", "app"]] | None = Field(
        default_factory=_app_only_visibility
    )


class AvailableTermsTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.AVAILABLE_TERMS
    details: AvailableTermsDetails = Field(default_factory=AvailableTermsDetails)


class TermDefinitionsTool(BaseToolConfig):
    type: ToolTypes = ToolTypes.TERM_DEFINITIONS
    details: TermDefinitionsDetails = Field(default_factory=TermDefinitionsDetails)

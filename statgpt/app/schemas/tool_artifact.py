from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import ToolTypes

from .data_query_outcome import DataQueryMcpPayload
from .discovery_datasets import DiscoveryDatasetsEvalAttachment, DiscoveryDatasetsOutcome
from .file_rags import BaseRagState, DialRagState
from .query_builder import DataQueryEvalAttachment, QueryBuilderAgentState
from .tool_states import DeepResearchToolMessageState, FailedToolMessageState, ToolMessageState


class ToolArtifact(BaseModel):

    state: ToolMessageState = Field(description="The state of the tool.")

    @property
    def type(self) -> ToolTypes:
        """The type of the tool to which the artifact belongs."""
        return self.state.type


class FailedToolArtifact(ToolArtifact):
    state: FailedToolMessageState


class DeepResearchArtifact(ToolArtifact):
    """Carries whether the Deep Research turn delivered its final report, so the Supreme Agent
    can end the turn without repeating the report (the tool streams it to the user directly)."""

    state: DeepResearchToolMessageState


class DataQueryOutcome(BaseModel):
    """What one run of the data query pipeline produced, before either interface renders it.

    The LangChain tool turns it into the tool message text plus a `DataQueryArtifact`; the MCP tool
    turns it into content blocks and structured content. Neither framework leaks in here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    response: str = Field(
        description="The pipeline's human-readable response, without the discovery datasets block."
    )
    data_responses: dict[str, DataResponse] = Field(
        default_factory=dict,
        description="Mapping from dataset id to response "
        "if the data request was successfully built and executed.",
    )
    state: QueryBuilderAgentState = Field(default_factory=QueryBuilderAgentState)
    mcp_payload: DataQueryMcpPayload = Field(default_factory=DataQueryMcpPayload)
    eval_attachment: DataQueryEvalAttachment = Field(default_factory=DataQueryEvalAttachment)
    discovery: DiscoveryDatasetsOutcome | None = Field(
        default=None,
        description="What the discovery datasets lookup did on this call, or `None` when the"
        " lookup is not configured for the channel.",
    )

    @property
    def discovery_block(self) -> str | None:
        """The rendered discovery datasets block, or `None` when there was nothing to show."""
        return self.discovery.rendered if self.discovery is not None else None


class DataQueryArtifact(ToolArtifact):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: QueryBuilderAgentState = Field(description="The state of the tool.")
    data_responses: dict[str, DataResponse] = Field(
        description="Mapping from dataset id to response "
        "if the data request was successfully built and executed."
    )
    mcp_payload: DataQueryMcpPayload = Field(
        default_factory=DataQueryMcpPayload,
        description="MCP-response-only data (not persisted to the tool state).",
    )
    eval_attachment: DataQueryEvalAttachment = Field(
        description="Attachment containing additional information for evaluation."
    )
    discovery_datasets_eval_attachment: DiscoveryDatasetsEvalAttachment | None = Field(
        default=None,
        description=(
            "What the discovery datasets lookup did on this call, for evaluation. `None` when"
            " the lookup is not configured for the channel."
        ),
    )

    @classmethod
    def from_outcome(cls, outcome: DataQueryOutcome) -> Self:
        discovery = outcome.discovery
        return cls(
            state=outcome.state,
            data_responses=outcome.data_responses,
            mcp_payload=outcome.mcp_payload,
            eval_attachment=outcome.eval_attachment,
            discovery_datasets_eval_attachment=(
                discovery.eval_attachment if discovery is not None else None
            ),
        )


# ~~~~~~~~~~~~~ File RAG ~~~~~~~~~~~~~


class BaseFileRagArtifact(ToolArtifact):
    state: BaseRagState


class DialRagArtifact(BaseFileRagArtifact):
    state: DialRagState

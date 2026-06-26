from pydantic import BaseModel, ConfigDict, Field

from statgpt.common.data.base import DataResponse
from statgpt.common.schemas import ToolTypes

from .file_rags import BaseRagState, DialRagState
from .query_builder import DataQueryEvalAttachment, QueryBuilderAgentState
from .tool_states import FailedToolMessageState, ToolMessageState


class ToolArtifact(BaseModel):

    state: ToolMessageState = Field(description="The state of the tool.")

    @property
    def type(self) -> ToolTypes:
        """The type of the tool to which the artifact belongs."""
        return self.state.type


class FailedToolArtifact(ToolArtifact):
    state: FailedToolMessageState


class DataQueryArtifact(ToolArtifact):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: QueryBuilderAgentState = Field(description="The state of the tool.")
    data_responses: dict[str, DataResponse] = Field(
        description="Mapping from dataset id to response "
        "if the data request was successfully built and executed."
    )
    eval_attachment: DataQueryEvalAttachment = Field(
        description="Attachment containing additional information for evaluation."
    )


class SdmxQueryAppArtifact(ToolArtifact):
    """Carries the upstream HTTP metadata for the SDMX query-app passthrough tool so the MCP
    provider can expose the status code and content type to the client (the raw body is returned
    as the tool's text content)."""

    status_code: int = Field(description="HTTP status code returned by the upstream request.")
    content_type: str | None = Field(
        default=None,
        description="Value of the upstream `Content-Type` response header, if present.",
    )


# ~~~~~~~~~~~~~ File RAG ~~~~~~~~~~~~~


class BaseFileRagArtifact(ToolArtifact):
    state: BaseRagState


class DialRagArtifact(BaseFileRagArtifact):
    state: DialRagState

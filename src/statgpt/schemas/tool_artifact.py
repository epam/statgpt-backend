from pydantic import BaseModel, ConfigDict, Field

from common.data.base import DataResponse
from common.schemas import ToolTypes

from .file_rags import BaseRagState, DialRagState
from .query_builder import QueryBuilderAgentState
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


# ~~~~~~~~~~~~~ File RAG ~~~~~~~~~~~~~


class BaseFileRagArtifact(ToolArtifact):
    state: BaseRagState


class DialRagArtifact(BaseFileRagArtifact):
    state: DialRagState

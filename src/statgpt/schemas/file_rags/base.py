from common.schemas import RAGVersion, ToolTypes
from statgpt.schemas.tool_states import ToolMessageState


class BaseRagState(ToolMessageState):
    type: ToolTypes = ToolTypes.FILE_RAG
    version: RAGVersion

    response: str  # This is not needed since we have content field
    answered_by: str

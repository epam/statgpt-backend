from statgpt.app.schemas.tool_states import ToolMessageState
from statgpt.common.schemas import RAGVersion, ToolTypes


class BaseRagState(ToolMessageState):
    type: ToolTypes = ToolTypes.FILE_RAG
    version: RAGVersion = RAGVersion.DIAL

    response: str = ""  # This is not needed since we have content field
    answered_by: str = ""

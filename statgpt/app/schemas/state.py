from pydantic import BaseModel, StrictStr


class ChatState(BaseModel):
    # TODO: add other fields to state model
    visualization_raw_llm_response: StrictStr | None = None

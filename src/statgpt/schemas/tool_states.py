from pydantic import BaseModel, Field

from common.schemas import ToolTypes


class ToolMessageState(BaseModel):
    type: ToolTypes = Field()


class FailedToolMessageState(ToolMessageState):
    error: str = Field(description="Error message from the tool")

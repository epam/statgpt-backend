from enum import StrEnum

from pydantic import BaseModel, Field

from common.schemas import ToolTypes


class ToolResponseStatus(StrEnum):
    # same as in langchain
    SUCCESS = "success"
    ERROR = "error"


class ToolMessageState(BaseModel):
    type: ToolTypes = Field()


class FailedToolMessageState(ToolMessageState):
    error: str = Field(description="Error message from the tool")

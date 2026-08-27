from enum import StrEnum

from pydantic import BaseModel, Field

from statgpt.common.schemas import ToolTypes


class ToolResponseStatus(StrEnum):
    # same as in langchain
    SUCCESS = "success"
    ERROR = "error"


class ToolMessageState(BaseModel):
    type: ToolTypes = Field()


class FailedToolMessageState(ToolMessageState):
    error: str = Field(description="Error message from the tool")


class DeepResearchToolMessageState(ToolMessageState):
    report_delivered: bool = Field(
        default=False,
        description=(
            "True when Deep Research delivered its final report on this turn (research complete)."
            " The report is streamed to the user by the tool, so the Supreme Agent must end the turn"
            " without repeating it. False for clarifying questions / plan-for-approval, which the"
            " Supreme Agent mediates."
        ),
    )

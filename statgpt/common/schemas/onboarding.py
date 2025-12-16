from typing import Literal

from pydantic import BaseModel, Field

from .base import BaseYamlModel
from .query import JsonQuery


class PredefinedTextResponse(BaseModel):

    response_type: Literal["predefined_text"] = Field(
        default="predefined_text",
        description="The type of the response, which is always 'predefined_text' for this class.",
    )
    text: str = Field(description="The predefined text response in markdown format.")


class PredefinedDataQueryResponse(BaseModel):

    response_type: Literal["predefined_data_query"] = Field(
        default="predefined_data_query",
        description="The type of the response, which is always 'predefined_data_query' for this class.",
    )
    text: str = Field(description="Text response in markdown format.")
    query: JsonQuery = Field(description="A predefined data query to be executed.")


Response = PredefinedTextResponse | PredefinedDataQueryResponse


class OnboardingTopic(BaseYamlModel):

    id: str = Field(description="The id of the onboarding topic.")
    short_title: str = Field(description="A short title for the topic.")
    question: str = Field(description="A question that triggers this topic during onboarding.")
    response: Response = Field(
        description="The response associated with the topic, which can be either a predefined text or a predefined data query.",
        discriminator="response_type",
    )
    subtopics: dict[str, "OnboardingTopic"] = Field(
        default_factory=dict,
        description="A dictionary of subtopics, where the key is a unique identifier for the subtopic.",
    )


class OnboardingConfig(BaseYamlModel):

    intro_message: str = Field(
        description="The introductory message sent to the user when they start a conversation."
    )
    completion_button_title: str = Field(
        default="Complete",
        description="The title of the button that the user clicks to complete the onboarding process.",
    )
    completion_button_text: str = Field(
        default="Complete",
        description="The text displayed in the chat as user message for the button click.",
    )
    completion_message: str = Field(
        description="The message sent to the user upon completion of the onboarding process."
    )
    topic_selection_prompt: str = Field(
        description="Prompt shown at root level when selecting topics."
    )
    topics: dict[str, OnboardingTopic] = Field(
        description="A dictionary of onboarding topics, where the key is a unique identifier for the topic."
    )

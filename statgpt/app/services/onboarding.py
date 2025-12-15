from __future__ import annotations

import logging
from typing import Literal

from aidial_sdk.chat_completion import Button, Choice, FormMetaclass
from aidial_sdk.pydantic.v2 import ConfigDict as DialConfigDict
from aidial_sdk.pydantic.v2 import Field as DialField
from pydantic import BaseModel, Field

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas.onboarding import (
    OnboardingConfig,
    OnboardingTopic,
    PredefinedTextResponse,
    Response,
)

from .chat_facade import ChannelServiceFacade
from .onboarding_response_appenders import ResponseAppenderFactory

_log = logging.getLogger(__name__)


class OnboardingState(BaseModel):
    """Tracks the current state of the onboarding process."""

    current_step: Literal["topic_selection", "completion_button", "completed"] = Field(
        default="topic_selection",
        description="The current step in the onboarding flow",
    )
    visited_topics: set[str] = Field(
        default_factory=set,
        description="Set of top-level topic IDs that have been explored",
    )
    current_path: list[str] = Field(
        default_factory=list,
        description="Path through the topic tree, e.g., ['topic_a', 'subtopic_1', 'sub_subtopic_2']",
    )

    def is_topic_selection(self) -> bool:
        return self.current_step == "topic_selection"

    def is_completed(self) -> bool:
        return self.current_step == "completed"

    def is_completion_button(self) -> bool:
        return self.current_step == "completion_button"

    def set_topic_selection(self) -> None:
        self.current_step = "topic_selection"

    def set_completed(self) -> None:
        self.current_step = "completed"
        self.current_path = []

    def set_completion_button(self) -> None:
        self.current_step = "completion_button"
        self.current_path = []


class CompletedSchema(BaseModel, metaclass=FormMetaclass):
    model_config = DialConfigDict(chat_message_input_disabled=True)


class OnboardingService:
    """
    Service for managing the onboarding flow with generic nested topic navigation.

    The flow follows this pattern:
    1. Show top-level topics (with intro message on first view)
    2. User clicks a topic -> navigate deeper into the tree
    3. Continue navigating until reaching a leaf node (topic with no subtopics)
    4. Show leaf node response
    5. Mark the top-level topic as visited
    6. Return to top-level topic selection
    7. When all top-level topics visited -> show completion
    """

    def __init__(self, config: OnboardingConfig):
        self.config = config

    def get_initial_form_schema(self, state: OnboardingState | None = None) -> dict:
        """
        Generate the initial form schema showing top-level topics with intro message.

        Args:
            state: Optional state to show progress. If None, creates new state.
        """
        if state is None:
            state = OnboardingState()

        _log.info("Generating initial onboarding form")
        return self._create_navigation_form(state, show_intro=True)

    def get_form_schema(
        self, state: OnboardingState, button_clicked: str | None = None
    ) -> dict | None:
        """
        Generate the next form schema based on current state and button clicked.

        This method handles generic tree navigation:
        - Adds button_clicked to current_path
        - If the topic at that path has subtopics, shows them
        - If the topic is a leaf (no subtopics), marks top-level topic as visited and returns to root

        Args:
            state: Current onboarding state
            button_clicked: ID of the button that was clicked

        Returns:
            Form schema dict, or None if onboarding is completed
        """
        _log.info(
            f"Generating form for state: {state.current_step}, button: {button_clicked}, path: {state.current_path}"
        )

        if button_clicked == "complete" and state.is_completion_button():
            # User clicked the final completion button
            state.set_completed()
            return CompletedSchema.model_json_schema()

        if state.is_completed():
            return CompletedSchema.model_json_schema()

        if not button_clicked:
            # No button clicked, show current state
            return self._create_navigation_form(state)

        # Add button to path and navigate
        state.current_path.append(button_clicked)

        # Get the topic at the current path
        current_topic = self._get_topic_at_path(state.current_path)

        if current_topic is None:
            _log.error(f"Topic not found at path: {state.current_path}")
            # Reset to root
            state.current_path = []
            return self._create_navigation_form(state)

        # Check if this is a leaf node (no subtopics)
        if not current_topic.subtopics:
            # Leaf node reached - mark top-level topic as visited
            if state.current_path:
                top_level_topic_id = state.current_path[0]
                state.visited_topics.add(top_level_topic_id)
                _log.info(f"Marked topic {top_level_topic_id} as visited")

            # Check if all topics have been visited
            if len(state.visited_topics) == len(self.config.topics):
                state.set_completion_button()
                return self._create_completion_button_form()

            # Reset path and return to top-level selection
            state.current_path = []
            return self._create_navigation_form(state)

        # Not a leaf node - show subtopics
        return self._create_navigation_form(state)

    def _get_topic_at_path(self, path: list[str]) -> OnboardingTopic | None:
        """
        Navigate through the topic tree following the given path.

        Args:
            path: List of topic IDs representing the path, e.g., ['topic_a', 'subtopic_1']

        Returns:
            The OnboardingTopic at that path, or None if not found
        """
        if not path:
            return None

        # Start with top-level topics
        current_topic = self.config.topics.get(path[0])
        if current_topic is None:
            return None

        # Navigate through the path
        for topic_id in path[1:]:
            if not current_topic.subtopics:
                return None
            current_topic = current_topic.subtopics.get(topic_id)
            if current_topic is None:
                return None

        return current_topic

    def _create_navigation_form(self, state: OnboardingState, show_intro: bool = False) -> dict:
        """
        Create a navigation form showing available topics at the current path level.

        Args:
            state: Current onboarding state
            show_intro: If True, prepends intro message to description
        """
        # Determine which topics to show
        if not state.current_path:
            # At root level - show unvisited top-level topics
            available_topics = {
                topic_id: topic
                for topic_id, topic in self.config.topics.items()
                if topic_id not in state.visited_topics
            }
        else:
            # Navigate to current location and show its subtopics
            parent_topic = self._get_topic_at_path(state.current_path)
            if parent_topic is None or not parent_topic.subtopics:
                # Invalid state, reset to root
                _log.warning(f"Invalid path {state.current_path}, resetting to root")
                state.current_path = []
                return self._create_navigation_form(state, show_intro=show_intro)
            available_topics = parent_topic.subtopics

        # Build buttons
        buttons = []
        for topic_id, topic in available_topics.items():
            buttons.append(
                Button(
                    const=topic_id,
                    submit=True,
                    title=topic.short_title,
                    populateText=topic.question,
                )
            )

        # Build description
        description_parts = []

        if show_intro and self.config.intro_message:
            description_parts.append(self.config.intro_message)
            description_parts.append("")  # Empty line

        # At root level, show main prompt
        if not state.current_path:
            description_parts.append(self.config.topic_selection_prompt)

        description = "\n".join(description_parts)

        class NavigationForm(BaseModel, metaclass=FormMetaclass):
            model_config = DialConfigDict(chat_message_input_disabled=True)

            choice: str | None = DialField(
                description=description,
                buttons=buttons,
            )

        return NavigationForm.model_json_schema()

    def _create_completion_button_form(self) -> dict:
        """Create completion form shown when all topics have been explored."""

        class CompletionButtonForm(BaseModel, metaclass=FormMetaclass):
            model_config = DialConfigDict(chat_message_input_disabled=True)

            completion: int | None = DialField(
                buttons=[
                    Button(
                        const="complete",
                        submit=True,
                        title=self.config.completion_button_title,
                        populateText=self.config.completion_button_text,
                    )
                ],
            )

        return CompletionButtonForm.model_json_schema()

    def _create_completion_form(self) -> dict:
        """
        Create the final completion form after all topics have been explored.
        This form simply disables input.
        """

        class CompletionForm(BaseModel, metaclass=FormMetaclass):
            model_config = DialConfigDict(chat_message_input_disabled=True)

            completion: int | None = DialField(
                description=self.config.completion_message,
                buttons=[],
            )

        return CompletionForm.model_json_schema()

    def get_response_for_path(self, path: list[str]) -> Response | None:
        """
        Get the response content for a topic at the given path.

        Args:
            path: Path through the topic tree, e.g., ['topic_a', 'subtopic_1']

        Returns:
            Response object (PredefinedTextResponse or PredefinedDataQueryResponse)
        """
        if len(path) == 1 and path[0] == "complete":
            # Special case for completion button
            return PredefinedTextResponse(text=self.config.completion_message)

        topic = self._get_topic_at_path(path)
        if topic is None:
            _log.error(f"Topic not found at path: {path}")
            return None

        return topic.response

    async def set_content(
        self,
        state: OnboardingState,
        button_clicked: str | None,
        choice: Choice,
        channel_service: ChannelServiceFacade,
        auth_context: AuthContext,
    ) -> None:
        _log.info(
            f"Processing onboarding: state={state.model_dump()}, button_clicked={button_clicked}"
        )

        # If we're navigating to a leaf node, get and append the response content
        if button_clicked:
            # Build the path after this click to check if it's a leaf
            temp_path = state.current_path + [button_clicked]
            response = self.get_response_for_path(temp_path)

            appender = ResponseAppenderFactory.get_appender(response)
            await appender.append_to_response(
                choice=choice,
                channel_service=channel_service,
                auth_context=auth_context,
            )

        # Get the next form schema based on current state and button click
        form_schema = self.get_form_schema(state, button_clicked)

        # Set form schema if onboarding is not completed
        if form_schema:
            choice.set_form_schema(form_schema)

        # Save the updated state
        # Convert sets to lists for JSON serialization
        state_dict = state.model_dump()
        state_dict["visited_topics"] = list(state_dict["visited_topics"])
        choice.set_state(state_dict)

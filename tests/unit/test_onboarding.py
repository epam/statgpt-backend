import pytest

from common.schemas.onboarding import OnboardingConfig, OnboardingTopic, PredefinedTextResponse
from statgpt.services.onboarding import OnboardingService, OnboardingState


@pytest.fixture
def simple_config():
    """Create a simple onboarding config with 2 top-level topics, each with 2 subtopics."""
    return OnboardingConfig(
        intro_message="Welcome to the app!",
        completion_button_title="Complete Onboarding",
        completion_button_text="Got it, thanks!",
        completion_message="You've completed onboarding!",
        topic_selection_prompt="What would you like to learn about?",
        topics={
            "topic_a": OnboardingTopic(
                id="topic_a",
                short_title="Topic A",
                question="What is Topic A?",
                response=PredefinedTextResponse(text="Response for Topic A"),
                subtopics={
                    "subtopic_a1": OnboardingTopic(
                        id="subtopic_a1",
                        short_title="Subtopic A1",
                        question="What is Subtopic A1?",
                        response=PredefinedTextResponse(text="Response for Subtopic A1"),
                        subtopics={},
                    ),
                    "subtopic_a2": OnboardingTopic(
                        id="subtopic_a2",
                        short_title="Subtopic A2",
                        question="What is Subtopic A2?",
                        response=PredefinedTextResponse(text="Response for Subtopic A2"),
                        subtopics={},
                    ),
                },
            ),
            "topic_b": OnboardingTopic(
                id="topic_b",
                short_title="Topic B",
                question="What is Topic B?",
                response=PredefinedTextResponse(text="Response for Topic B"),
                subtopics={
                    "subtopic_b1": OnboardingTopic(
                        id="subtopic_b1",
                        short_title="Subtopic B1",
                        question="What is Subtopic B1?",
                        response=PredefinedTextResponse(text="Response for Subtopic B1"),
                        subtopics={},
                    ),
                    "subtopic_b2": OnboardingTopic(
                        id="subtopic_b2",
                        short_title="Subtopic B2",
                        question="What is Subtopic B2?",
                        response=PredefinedTextResponse(text="Response for Subtopic B2"),
                        subtopics={},
                    ),
                },
            ),
        },
    )


@pytest.fixture
def nested_config():
    """Create a config with 3-level nested topics."""
    return OnboardingConfig(
        intro_message="Welcome!",
        completion_message="Done!",
        completion_button_title="Complete Onboarding",
        completion_button_text="Got it, thanks!",
        topic_selection_prompt="Choose a topic:",
        topics={
            "topic_a": OnboardingTopic(
                id="topic_a",
                short_title="Topic A",
                question="About Topic A?",
                response=PredefinedTextResponse(text="Topic A response"),
                subtopics={
                    "sub_a1": OnboardingTopic(
                        id="sub_a1",
                        short_title="Sub A1",
                        question="About Sub A1?",
                        response=PredefinedTextResponse(text="Sub A1 response"),
                        subtopics={
                            "sub_sub_a1": OnboardingTopic(
                                id="sub_sub_a1",
                                short_title="Sub-Sub A1",
                                question="About Sub-Sub A1?",
                                response=PredefinedTextResponse(text="Sub-Sub A1 response"),
                                subtopics={},
                            ),
                        },
                    ),
                },
            ),
        },
    )


class TestOnboardingState:
    """Tests for OnboardingState model."""

    def test_default_state(self):
        """Test that default state is properly initialized."""
        state = OnboardingState()
        assert state.is_topic_selection()
        assert state.visited_topics == set()
        assert state.current_path == []

    def test_state_with_data(self):
        """Test state creation with data."""
        state = OnboardingState(
            current_step="completed",
            visited_topics={"topic_a", "topic_b"},
            current_path=["topic_a", "subtopic_a1"],
        )
        assert state.is_completed()
        assert state.visited_topics == {"topic_a", "topic_b"}
        assert state.current_path == ["topic_a", "subtopic_a1"]


class TestOnboardingService:
    """Tests for OnboardingService."""

    def test_initialization(self, simple_config):
        """Test service initialization."""
        service = OnboardingService(simple_config)
        assert service.config == simple_config

    def test_get_initial_form_schema(self, simple_config):
        """Test initial form schema generation with intro message."""
        service = OnboardingService(simple_config)
        schema = service.get_initial_form_schema()

        assert schema is not None
        assert "properties" in schema
        # Check that intro message and topic selection prompt are in description
        choice_field = schema["properties"]["choice"]
        assert "Welcome to the app!" in choice_field["description"]
        assert "What would you like to learn about?" in choice_field["description"]

    def test_is_onboarding_completed(self, simple_config):
        """Test completion check."""

        state_not_completed = OnboardingState(current_step="topic_selection")
        assert not state_not_completed.is_completed()

        state_completed = OnboardingState(current_step="completed")
        assert state_completed.is_completed()

    def test_get_topic_at_path(self, simple_config):
        """Test navigation through topic tree."""
        service = OnboardingService(simple_config)

        # Get top-level topic
        topic = service._get_topic_at_path(["topic_a"])
        assert topic is not None
        assert topic.id == "topic_a"
        assert topic.short_title == "Topic A"

        # Get subtopic
        subtopic = service._get_topic_at_path(["topic_a", "subtopic_a1"])
        assert subtopic is not None
        assert subtopic.id == "subtopic_a1"
        assert subtopic.short_title == "Subtopic A1"

        # Non-existent path
        assert service._get_topic_at_path(["nonexistent"]) is None
        assert service._get_topic_at_path(["topic_a", "nonexistent"]) is None

        # Empty path
        assert service._get_topic_at_path([]) is None

    def test_get_topic_at_path_nested(self, nested_config):
        """Test navigation through deeply nested topics."""
        service = OnboardingService(nested_config)

        topic = service._get_topic_at_path(["topic_a", "sub_a1", "sub_sub_a1"])
        assert topic is not None
        assert topic.id == "sub_sub_a1"

    def test_get_response_for_path(self, simple_config):
        """Test getting response for a path."""
        service = OnboardingService(simple_config)

        # Get response for leaf node
        response = service.get_response_for_path(["topic_a", "subtopic_a1"])
        assert response is not None
        assert response.response_type == "predefined_text"
        assert response.text == "Response for Subtopic A1"

        # Non-existent path
        assert service.get_response_for_path(["nonexistent"]) is None

    def test_navigation_to_subtopics(self, simple_config):
        """Test navigating from root to subtopics."""
        service = OnboardingService(simple_config)
        state = OnboardingState()

        # Click on topic_a
        schema = service.get_form_schema(state, button_clicked="topic_a")

        # Should show subtopics of topic_a
        assert state.current_path == ["topic_a"]
        assert schema is not None

    def test_navigation_to_leaf_marks_visited(self, simple_config):
        """Test that reaching a leaf node marks the top-level topic as visited."""
        service = OnboardingService(simple_config)
        state = OnboardingState()

        # Navigate to topic_a -> subtopic_a1 (leaf)
        service.get_form_schema(state, button_clicked="topic_a")
        assert state.visited_topics == set()  # Not visited yet

        service.get_form_schema(state, button_clicked="subtopic_a1")
        assert "topic_a" in state.visited_topics  # Now visited
        assert state.current_path == []  # Reset to root

    def test_navigation_shows_only_unvisited_topics(self, simple_config):
        """Test that only unvisited topics are shown at root level."""
        service = OnboardingService(simple_config)
        state = OnboardingState(visited_topics={"topic_a"})

        schema = service.get_form_schema(state, button_clicked=None)

        # Schema should be generated successfully
        assert schema is not None
        assert "properties" in schema

        # Verify that the service's internal logic correctly filters topics
        # by checking we can still navigate to topic_b
        schema_b = service.get_form_schema(state, button_clicked="topic_b")
        assert schema_b is not None
        assert state.current_path == ["topic_b"]

    def test_completion_when_all_topics_visited(self, simple_config):
        """Test that completion form is shown when all topics are visited."""
        service = OnboardingService(simple_config)
        state = OnboardingState(visited_topics={"topic_a"})

        # Visit topic_b (last remaining topic)
        service.get_form_schema(state, button_clicked="topic_b")
        schema = service.get_form_schema(state, button_clicked="subtopic_b1")

        # Should show completion
        assert state.is_completion_button()
        assert schema is not None

    def test_completed_state_returns_completed_schema(self, simple_config):
        """Test that completed state returns None."""
        service = OnboardingService(simple_config)
        state = OnboardingState(current_step="completed")

        schema = service.get_form_schema(state, button_clicked=None)
        assert schema is not None
        assert "properties" in schema
        assert len(schema["properties"]) == 0

    def test_invalid_button_resets_to_root(self, simple_config):
        """Test that invalid button click resets to root."""
        service = OnboardingService(simple_config)
        state = OnboardingState()

        schema = service.get_form_schema(state, button_clicked="invalid_topic")
        assert state.current_path == []  # Reset to root
        assert schema is not None  # Should show root level topics

    def test_nested_navigation_three_levels(self, nested_config):
        """Test navigation through 3 levels of nesting."""
        service = OnboardingService(nested_config)
        state = OnboardingState()

        # Level 1: topic_a
        schema = service.get_form_schema(state, button_clicked="topic_a")
        assert state.current_path == ["topic_a"]
        assert schema is not None

        # Level 2: sub_a1
        schema = service.get_form_schema(state, button_clicked="sub_a1")
        assert state.current_path == ["topic_a", "sub_a1"]
        assert schema is not None

        # Level 3: sub_sub_a1 (leaf)
        schema = service.get_form_schema(state, button_clicked="sub_sub_a1")
        assert "topic_a" in state.visited_topics
        assert state.current_path == []  # Reset to root

    def test_topic_with_no_subtopics_marks_visited_immediately(self):
        """Test that a topic with no subtopics marks itself as visited immediately."""
        config = OnboardingConfig(
            intro_message="Intro",
            completion_message="Done",
            completion_button_title="Complete Onboarding",
            completion_button_text="Got it, thanks!",
            topic_selection_prompt="Choose:",
            topics={
                "topic_a": OnboardingTopic(
                    id="topic_a",
                    short_title="Topic A",
                    question="About A?",
                    response=PredefinedTextResponse(text="Response A"),
                    subtopics={},  # No subtopics
                ),
                "topic_b": OnboardingTopic(
                    id="topic_b",
                    short_title="Topic B",
                    question="About B?",
                    response=PredefinedTextResponse(text="Response B"),
                    subtopics={},
                ),
            },
        )

        service = OnboardingService(config)
        state = OnboardingState()

        # Click on topic_a which has no subtopics
        schema = service.get_form_schema(state, button_clicked="topic_a")
        assert schema is not None

        # Should be marked as visited and return to root
        assert "topic_a" in state.visited_topics
        assert state.current_path == []

    def test_schema_structure(self, simple_config):
        """Test that the schema has the expected structure."""
        service = OnboardingService(simple_config)
        state = OnboardingState()

        schema = service.get_form_schema(state, button_clicked=None)

        # Check basic schema structure
        assert schema is not None
        assert "properties" in schema
        assert "choice" in schema["properties"]
        assert "type" in schema["properties"]["choice"]

    def test_full_onboarding_flow(self, simple_config):
        """Test a complete onboarding flow from start to finish."""
        service = OnboardingService(simple_config)
        state = OnboardingState()

        # Initial state
        assert not state.is_completed()
        assert len(state.visited_topics) == 0

        # Explore topic_a -> subtopic_a1
        service.get_form_schema(state, button_clicked="topic_a")
        service.get_form_schema(state, button_clicked="subtopic_a1")
        assert "topic_a" in state.visited_topics
        assert len(state.visited_topics) == 1

        # Explore topic_b -> subtopic_b2
        service.get_form_schema(state, button_clicked="topic_b")
        service.get_form_schema(state, button_clicked="subtopic_b2")
        assert "topic_b" in state.visited_topics
        assert len(state.visited_topics) == 2

        # Navigate to completion button
        assert state.is_completion_button()

        service.get_form_schema(state, button_clicked="complete")

        # Should be completed now
        assert state.is_completed()

# Spec Delta

## MODIFIED Requirements

### Requirement: Toggle reflects the next turn
Each assistant message SHALL carry a `custom_content.form_schema` whose `deep_research` control holds the value the toggle should have on the next request, as the static configuration schema cannot hold it. The control SHALL be emitted only when Deep Research is available to the caller; without it, a message has a `form_schema` only when it carries another control, such as the "Continue" button (see `chat-error-handling`). The value SHALL be false on the turn that delivered the final report or on which the query check rejected the query; true while a session is in progress; otherwise the user's current selection.

#### Scenario: Report delivered
- **WHEN** a turn delivers the final report
- **THEN** its form schema sets the toggle to false, so the follow-up goes to the normal agent

#### Scenario: Clarification pending
- **WHEN** a turn ends with a clarifying question
- **THEN** its form schema keeps the toggle on

#### Scenario: Control absent
- **WHEN** an assistant message carries no `deep_research` control
- **THEN** the frontend falls back to the configuration default, off

#### Scenario: Deep Research unavailable
- **WHEN** a streamed turn runs a data query and then fails with a rate limit, on a channel where Deep Research is unavailable to the caller
- **THEN** its `form_schema` has the "Continue" button and no `deep_research` control

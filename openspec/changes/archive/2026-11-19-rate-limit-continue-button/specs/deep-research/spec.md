# Spec Delta

## MODIFIED Requirements

### Requirement: Deployment failures end the turn with Deep Research's message
When the Deep Research deployment fails, before or during the stream, the turn SHALL end with a DIAL error of status 500 carrying Deep Research's `display_message`, read from the stream's error object or from `error` in an HTTP error body. Its internal `message` SHALL never be shown; without a usable `display_message` the text SHALL be "Deep Research could not complete this request.". It SHALL NOT tell the user to try again, as the Continue note warns against the "Try again" button. Its status and `code` SHALL NOT be interpreted (e.g. as a rate limit). On a streamed turn the Continue note (see `chat-error-handling`) SHALL follow that text.

#### Scenario: Error with a display message
- **WHEN** Deep Research fails with `display_message` "Research failed (ref 7f3a). Please retry." on a non-streaming request
- **THEN** the response is only a DIAL error with that text

#### Scenario: Error without a display message
- **WHEN** Deep Research fails without a usable `display_message` on a streamed request
- **THEN** the user gets "Deep Research could not complete this request.", a blank line and the Continue note

#### Scenario: Error with a display message while streaming
- **WHEN** Deep Research fails with `display_message` "Research failed (ref 7f3a). Please retry." on a streamed request
- **THEN** the user gets a DIAL error with that text followed by a blank line and the Continue note, and the conversation state is still saved, including the failed call and its notes in the tool messages

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
- **WHEN** a streamed turn fails with a rate limit on a channel where Deep Research is unavailable to the caller
- **THEN** its `form_schema` has the "Continue" button and no `deep_research` control

### Requirement: Mediation loop ends
The mediation loop SHALL end when the agent writes to the user or Deep Research delivers its final report. It SHALL also stop after 50 iterations, set far above any realistic number of clarification rounds, as a guard against a deployment that never finishes or an agent stuck producing invalid responses. On reaching the cap, the standard failure text SHALL be appended to the answer once and the session SHALL be kept, so the next turn with the toggle on resumes it.

#### Scenario: Cap reached
- **WHEN** 50 mediation iterations pass without the agent writing to the user or a report being delivered
- **THEN** the answer ends with "Deep Research could not complete this request." and the session stays in the state

### Requirement: Other failures append the standard text
A failure of a Deep Research tool call other than a deployment error SHALL be recorded as an error tool response, and the turn SHALL end with "Deep Research could not complete this request." appended to the answer once.

#### Scenario: Unexpected exception
- **WHEN** the tool raises an exception that is not a deployment error
- **THEN** the answer ends with the standard failure text and no DIAL error is raised

## ADDED Requirements

### Requirement: Continue button after a deployment failure
A streamed turn that ends with a Deep Research deployment failure SHALL carry the same "Continue" button as a rate-limited turn (see `chat-error-handling`), next to the `deep_research` control. It SHALL carry the button whether or not a session was kept, because the failed call and its error stay in the history for the next turn. A non-streaming turn SHALL NOT carry the button.

#### Scenario: Failure with a session kept
- **WHEN** Deep Research sends a state and then fails during the stream of a streamed turn
- **THEN** the response carries a `form_schema` with the "Continue" button and the `deep_research` control on, then the DIAL error

#### Scenario: Failure with no session
- **WHEN** the Deep Research deployment cannot be reached on the first call of a session, in a streamed turn
- **THEN** the response still carries the "Continue" button, and on "Continue" the agent sees the failed call in the history while a new session is started

#### Scenario: Other Deep Research failure
- **WHEN** a Deep Research tool call fails with an exception that is not a deployment error
- **THEN** the answer ends with the standard failure text and has no "Continue" button

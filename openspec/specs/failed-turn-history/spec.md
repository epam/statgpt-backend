# failed-turn-history Specification

## Purpose

Shows the Supreme Agent, on later turns, which earlier turns failed and what the user was shown, so that it neither repeats a failed turn's empty answer nor takes the part of an answer an error stopped for a complete one.

## Requirements

### Requirement: A failed turn records its error in the state
`ChannelCompletion` SHALL set `error` in the state of a turn that fails, and SHALL set it to `None` on a turn that succeeds. A turn that ends with a DIAL error raised to the client (a Deep Research failure, a rate limit, a forbidden deployment) SHALL also save the error's `display_message` as `error_display_message`.

#### Scenario: Rate limit
- **WHEN** a turn fails with an `openai.RateLimitError`
- **THEN** its state holds the raw error as `error`, and the message shown to the user as `error_display_message`

#### Scenario: Successful turn
- **WHEN** a turn completes without an error
- **THEN** its state holds `error` = `None`

### Requirement: The agent sees only the error shown to the user
The notes about a failed turn SHALL quote `error_display_message` only, never the raw `error`, which can carry internal details such as cost limits.

#### Scenario: Raw error kept out of the prompt
- **WHEN** a failed turn's `error` is `Error code: 429 - Hit cost rate limit. Day limit: $10.30 / $10.00` and its `error_display_message` is `You've exceeded your token limit. Please try again in 14 hours 30 minutes`
- **THEN** the note quotes the display message, and the cost limit appears nowhere in the agent's prompt

### Requirement: A turn that failed before any answer is shown as a note
When the history is built with tool messages, an assistant message with no content and no tool calls SHALL be replaced by a system note if its state has `error` set, or if it has no state (the turn was cut off before its state was sent, or the request was rejected). The note SHALL be `failedTurnWithErrorMessageToAgent`, quoting `error_display_message`, if the state holds one, and `failedTurnMessageToAgent` otherwise.

#### Scenario: Error shown to the user is known
- **WHEN** an earlier assistant message is empty and its state holds `error` and `error_display_message`
- **THEN** the agent sees, in its place, a system note saying no answer was given and quoting the error shown to the user

#### Scenario: Error shown to the user is not known
- **WHEN** an earlier assistant message is empty and has no state, or its state holds `error` but no `error_display_message`
- **THEN** the agent sees, in its place, a system note saying no answer was given, without an error

#### Scenario: Empty successful turn
- **WHEN** an earlier assistant message is empty and its state holds `error` = `None`
- **THEN** the agent sees it as an empty answer, with no note

### Requirement: A turn an error stopped mid-answer is marked as stopped
When the history is built with tool messages, an assistant message with content and no tool calls whose state holds `error` and `error_display_message` SHALL be kept as it is, and followed by the system note `interruptedTurnMessageToAgent`. The note SHALL say that an error stopped the response, that the user saw only the part above followed by the error, and SHALL quote `error_display_message`.

#### Scenario: Rate limit during streaming
- **WHEN** a turn streamed `US GDP grew in 2023, from` and then failed with a rate limit
- **THEN** on later turns the agent sees that text as its answer, followed by a system note saying the response was stopped by an error and quoting the error shown to the user

### Requirement: An answer that shows its own error has no note
An assistant message with content whose state holds `error` but no `error_display_message` SHALL be kept as it is, with no note: the error was shown in the message itself, e.g. `An error occurred while processing your request.` appended after an unexpected exception, or the explanation sent for a command-only message.

#### Scenario: Unexpected exception during streaming
- **WHEN** a turn streamed part of an answer, failed with an unexpected exception, and appended `An error occurred while processing your request.`
- **THEN** the agent sees the message as it was shown, with no note

### Requirement: A stopped response is kept as it is
An assistant message with content and no state SHALL be kept as it is, with no note: a response the user stopped has no state.

#### Scenario: User pressed Stop
- **WHEN** an earlier assistant message has partial content and no state
- **THEN** the agent sees it as its answer, with no note

### Requirement: The failed turn's tool messages come before the note
The tool calls and responses a failed turn saved in its state SHALL be shown before the part of its answer that was shown and before its note, in the order they happened. A turn that fails in Deep Research saves them, so that later turns see the failed call; other failed turns save none.

#### Scenario: Deep Research call failed
- **WHEN** a turn's Deep Research call failed with a Deep Research error and no answer was shown
- **THEN** the agent sees the tool call, its error response, the note that the Deep Research call failed, and then the note quoting the error shown to the user

### Requirement: Guardrails see the history as it was sent
The history built without tool messages, as used by the out-of-scope check and the Deep Research query check, SHALL show every assistant message as it was sent, with no notes about failed turns.

#### Scenario: Out-of-scope check after a failed turn
- **WHEN** the out-of-scope check runs after a turn that failed with an empty answer
- **THEN** it sees that turn as an empty assistant message

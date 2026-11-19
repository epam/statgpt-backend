# Spec Delta

## MODIFIED Requirements

### Requirement: The agent sees only the error shown to the user
The notes about a failed turn SHALL quote `error_display_message` only, never the raw `error`, which can carry internal details such as cost limits.

#### Scenario: Raw error kept out of the prompt
- **WHEN** a failed turn's `error` is `Error code: 429 - Hit cost rate limit. Day limit: $10.30 / $10.00` and its `error_display_message` is `You've exceeded your token limit for the underlying deployment. The limit resets in 14 hours 30 minutes.` followed by the Continue note (see `chat-error-handling`)
- **THEN** the note quotes the display message, Continue note included, and the cost limit appears nowhere in the agent's prompt

### Requirement: A failed turn's note invites a retry
The notes `failedTurnMessageToAgent`, `failedTurnWithErrorMessageToAgent` and `interruptedTurnMessageToAgent` SHALL say that the error affected only that attempt and may have been resolved since, and SHALL tell the agent that if the user asks to continue or to retry, it answers the message now (gives the full response, for a turn stopped mid-answer) instead of repeating the error. The notes that quote an error SHALL say this holds even if the error said to wait or to try again later.

#### Scenario: Continue after an error that said to try again later
- **WHEN** a Deep Research turn failed, the next turn failed with `You've exceeded your daily token limit for the underlying deployment. The limit resets in 1 day.`, and the user then sends `continue` after the limit was lifted
- **THEN** the note after the failed turn tells the agent to answer the user's message now instead of repeating the error, so the agent resumes Deep Research rather than replying that it cannot continue for a day

#### Scenario: Two failed turns in a row
- **WHEN** two turns in a row failed before any answer, and the user then sends `continue`
- **THEN** the agent answers the user's earlier message rather than replying with an empty answer

### Requirement: The failed turn's tool messages come before the note
The tool calls and responses a failed turn saved in its state SHALL be shown before its answer and its note, in the order they happened. They are shown only when the history is built with tool messages. A turn that fails in Deep Research or with a rate-limit DIAL error saves them, so that later turns see the finished calls; other failed turns save none.

#### Scenario: Deep Research call failed
- **WHEN** a turn's Deep Research call failed with a Deep Research error and no answer was shown
- **THEN** the agent sees the tool call, its error response, the note that the Deep Research call failed, the empty answer, and then the note quoting the error shown to the user

#### Scenario: Rate limit after a tool call
- **WHEN** a turn ran a data query tool, streamed `US GDP grew in 2023, from` and then failed with a rate limit
- **THEN** on later turns the agent sees the tool call and its response, then that text as its answer, then the note saying the response was stopped by an error and quoting the error shown to the user

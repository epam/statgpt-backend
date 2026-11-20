# Spec Delta

## MODIFIED Requirements

### Requirement: Rate-limit message
The rate-limit `display_message` SHALL name the exceeded limits (minute, daily, weekly, monthly) when the deployment's message reveals them, and say when the limit resets, from `Retry-After` when present. It SHALL NOT tell the user to try again, as that would point at the "Try again" button the Continue note warns against. The error SHALL carry `underlying_deployment`, the deployment id taken from the failed request's URL, and the original error body. On a streamed turn the Continue note follows the message.

#### Scenario: Daily limit
- **WHEN** the Supreme Agent's model returns 429 with "Day limit: 120000 / 100000 tokens" and `Retry-After: 3700` on a non-streaming request
- **THEN** the user gets "You've exceeded your daily token limit for the underlying deployment. The limit resets in 1 hour 1 minute.", with HTTP status 500 and code "429"

#### Scenario: Daily limit while streaming
- **WHEN** the same rate limit happens on a streamed request
- **THEN** the user gets the same message followed by the Continue note

#### Scenario: No retry time
- **WHEN** the Supreme Agent's model returns 429 without `Retry-After` on a non-streaming request
- **THEN** the user gets "You've exceeded your token limit for the underlying deployment. Please wait for the limit to reset."

## ADDED Requirements

### Requirement: Continue button after a rate limit
A streamed turn that ends with a rate-limit DIAL error SHALL carry, in its message's `form_schema`, a `button` property with a single submit button titled "Continue". The button SHALL send the user message "Continue". The button SHALL NOT depend on Deep Research: where Deep Research is unavailable to the caller, the `form_schema` SHALL hold only the button; where the Deep Research toggle is shown, the same `form_schema` SHALL keep the `deep_research` control next to the button.

#### Scenario: Button next to the error
- **WHEN** a streamed turn streams part of an answer and then fails with a rate limit from the Supreme Agent's model
- **THEN** the response carries the partial answer, a `form_schema` with the "Continue" button, and then the rate-limit error

#### Scenario: Button and Deep Research toggle together
- **WHEN** a streamed turn on a channel with Deep Research available fails with a rate limit
- **THEN** its `form_schema` has both the `deep_research` control and the "Continue" button

#### Scenario: Channel without Deep Research
- **WHEN** a streamed turn fails with a rate limit on a channel with no `deep_research` tool
- **THEN** its `form_schema` has the "Continue" button and no `deep_research` control

### Requirement: No Continue button elsewhere
A non-streaming turn SHALL NOT carry the "Continue" button, as its response is only the error. A turn that succeeds, or ends with an error other than a rate limit or a Deep Research deployment failure (see `deep-research`), SHALL NOT carry it either.

#### Scenario: Non-streaming request
- **WHEN** a non-streaming turn fails with a rate limit from the Supreme Agent's model
- **THEN** the response is only the rate-limit error, with no button

#### Scenario: Successful turn
- **WHEN** a streamed turn completes without an error
- **THEN** its `form_schema`, if any, has no `button` property

#### Scenario: Forbidden deployment
- **WHEN** a streamed turn fails because the Supreme Agent's model returns 403
- **THEN** its `form_schema`, if any, has no `button` property

### Requirement: Continue note goes with the button
Every error `display_message` of a turn that carries the "Continue" button SHALL end with the Continue note, after a blank line. The note is this text: `Press "Continue" (or type "continue") to pick up from where the response stopped: what was already done is kept. "Try again" starts over from your last message and repeats that work, so the response takes longer and uses more tokens.`. A turn without the button SHALL NOT get the note.

#### Scenario: Rate limit while streaming
- **WHEN** a streamed turn fails with a daily rate limit and `Retry-After: 3700`
- **THEN** the user gets "You've exceeded your daily token limit for the underlying deployment. The limit resets in 1 hour 1 minute.", a blank line and the Continue note

#### Scenario: No note without the button
- **WHEN** a non-streaming turn fails with a rate limit
- **THEN** its `display_message` doesn't contain the Continue note

### Requirement: The saved error includes the Continue note
`error_display_message` in the state SHALL hold the full text the user was shown, Continue note included, so the failed-turn notes quote what the user saw.

#### Scenario: Note in the state
- **WHEN** a streamed turn fails with a rate limit
- **THEN** the state's `error_display_message` ends with the Continue note

### Requirement: A rate-limited turn saves its tool messages
A turn that fails with a rate-limit DIAL error SHALL save in its state the tool calls and responses it completed before the error, so a later turn can reuse them instead of calling the tools again.

#### Scenario: Rate limit after a data query
- **WHEN** a turn ran a data query tool and the Supreme Agent's model then hit a rate limit while writing the answer
- **THEN** the turn's state holds the tool call and its response, the failed message carries the query's data attachments (see `data-query-agent-response`), and on the next turn ("Continue") the agent sees the saved call and does not need to run the query again

#### Scenario: Rate limit after Deep Research replied
- **WHEN** in a Deep Research turn, Deep Research returned a clarifying question and the Supreme Agent's model then hit a rate limit before the question was shown
- **THEN** the turn's state holds the Deep Research call and its response next to the session, so on the next turn ("Continue") the agent shows the pending question instead of passing "Continue" on to Deep Research as the user's answer

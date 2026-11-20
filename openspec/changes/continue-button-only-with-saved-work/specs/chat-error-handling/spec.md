# Spec Delta

## MODIFIED Requirements

### Requirement: Rate-limit message
The rate-limit `display_message` SHALL name the exceeded limits (minute, daily, weekly, monthly) when the deployment's message reveals them, and say when the limit resets, from `Retry-After` when present. It SHALL NOT tell the user to try again, as that would point at the "Try again" button the Continue note warns against on a turn that carries the note. The error SHALL carry `underlying_deployment`, the deployment id taken from the failed request's URL, and the original error body. On a turn that carries the "Continue" button the Continue note follows the message.

#### Scenario: Daily limit
- **WHEN** the Supreme Agent's model returns 429 with "Day limit: 120000 / 100000 tokens" and `Retry-After: 3700` on a non-streaming request
- **THEN** the user gets "You've exceeded your daily token limit for the underlying deployment. The limit resets in 1 hour 1 minute.", with HTTP status 500 and code "429"

#### Scenario: Daily limit while streaming
- **WHEN** the same rate limit happens on a streamed request, after the turn ran a data query
- **THEN** the user gets the same message followed by the Continue note

#### Scenario: Daily limit while streaming, nothing to continue
- **WHEN** the same rate limit happens on a streamed request, on the first model call of the turn
- **THEN** the user gets the same message, with no Continue note

#### Scenario: No retry time
- **WHEN** the Supreme Agent's model returns 429 without `Retry-After` on a non-streaming request
- **THEN** the user gets "You've exceeded your token limit for the underlying deployment. Please wait for the limit to reset."

### Requirement: Continue button after a rate limit
A streamed turn that ends with a rate-limit DIAL error and has something to continue (see "A turn has something to continue") SHALL carry, in its message's `form_schema`, a `button` property with a single submit button titled "Continue". The button SHALL send the user message "Continue". The button SHALL NOT depend on Deep Research: where Deep Research is unavailable to the caller, the `form_schema` SHALL hold only the button; where the Deep Research toggle is shown, the same `form_schema` SHALL keep the `deep_research` control next to the button.

#### Scenario: Button next to the error
- **WHEN** a streamed turn streams part of an answer and then fails with a rate limit from the Supreme Agent's model
- **THEN** the response carries the partial answer, a `form_schema` with the "Continue" button, and then the rate-limit error

#### Scenario: Button and Deep Research toggle together
- **WHEN** a streamed turn on a channel with Deep Research available runs a data query and then fails with a rate limit
- **THEN** its `form_schema` has both the `deep_research` control and the "Continue" button

#### Scenario: Channel without Deep Research
- **WHEN** a streamed turn runs a data query and then fails with a rate limit, on a channel with no `deep_research` tool
- **THEN** its `form_schema` has the "Continue" button and no `deep_research` control

### Requirement: No Continue button elsewhere
A non-streaming turn SHALL NOT carry the "Continue" button, as its response is only the error. A turn that succeeds, or ends with an error other than a rate limit or a Deep Research deployment failure (see `deep-research`), SHALL NOT carry it either. A rate-limited turn with nothing to continue SHALL NOT carry it: "Try again" loses nothing there.

#### Scenario: Non-streaming request
- **WHEN** a non-streaming turn fails with a rate limit from the Supreme Agent's model
- **THEN** the response is only the rate-limit error, with no button

#### Scenario: Successful turn
- **WHEN** a streamed turn completes without an error
- **THEN** its `form_schema`, if any, has no `button` property

#### Scenario: Forbidden deployment
- **WHEN** a streamed turn fails because the Supreme Agent's model returns 403
- **THEN** its `form_schema`, if any, has no `button` property

#### Scenario: Rate limit before the agent did anything
- **WHEN** a streamed turn fails with a rate limit on the out-of-scope check or on the agent's first model call, before any tool call or content
- **THEN** its `form_schema`, if any, has no `button` property, and the error has no Continue note

#### Scenario: Only configured tool calls ran
- **WHEN** a streamed turn ran only the tool calls the channel configures to run before every turn, and then fails with a rate limit on the agent's first model call
- **THEN** its `form_schema`, if any, has no `button` property

#### Scenario: A "Continue" turn that fails at once
- **WHEN** a turn ran a data query and failed with a rate limit, the user pressed "Continue", and the new turn fails with a rate limit on its first model call
- **THEN** the new turn has no button and no note, while the earlier failed message keeps its state, so "Try again" sends "Continue" again and the agent still sees the saved query

## ADDED Requirements

### Requirement: A turn has something to continue
A failed turn SHALL have something to continue when it holds a tool call the agent made in this turn, with or without its response, or content already sent to the user in this turn: "Try again" would delete both with the failed message. Tool calls the channel configures to run before every turn SHALL NOT count. The rule SHALL NOT read the content of a tool response or of Deep Research's state.

#### Scenario: Tool call with an error response
- **WHEN** a streamed turn ran a data query that failed, and the agent's next model call hit a rate limit
- **THEN** the turn has something to continue and carries the "Continue" button

#### Scenario: Content only
- **WHEN** a streamed turn called no tool, streamed part of its answer and then hit a rate limit
- **THEN** the turn has something to continue and carries the "Continue" button

#### Scenario: Deep Research deployment failure
- **WHEN** a streamed turn ends with a Deep Research deployment failure
- **THEN** the turn has something to continue, as the failure happened inside a Deep Research call, and carries the "Continue" button (see `deep-research`)

#### Scenario: Nothing done
- **WHEN** a streamed turn hit a rate limit before any tool call of the agent and before any content
- **THEN** the turn has nothing to continue

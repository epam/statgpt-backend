# chat-error-handling Specification

## Purpose

Defines how a chat completion request to a channel reports failures: which ones become DIAL errors with which status, code and user-facing `display_message`, which ones are answered with a plain message instead, and what the response still carries when a turn fails. Deep Research's own failure rules belong to `deep-research`; how later turns show a failed turn to the agent belongs to `failed-turn-history`.

## Requirements

### Requirement: Unknown channel
A chat completion or configuration request for a deployment id with no channel SHALL be refused with status 404, code `deployment_not_found` and the message "The API deployment for this resource does not exist.".

#### Scenario: Missing channel
- **WHEN** a request arrives for deployment `missing` and no channel has that deployment id
- **THEN** the response is a 404 DIAL error with code `deployment_not_found`

### Requirement: Usage and state survive a DIAL error
When a turn fails with an error that is reported as a DIAL error after the turn started, the response SHALL first emit the token usage stage and the conversation state, and only then raise the error, so the usage is accounted and the next turn keeps its state. This is the desired state, not yet fully supported: a content-filter error and a DIAL error passed through unchanged are still raised at once, without usage or state.

#### Scenario: Rate limit mid-turn
- **WHEN** the Supreme Agent's model hits a rate limit after other model calls of the turn succeeded
- **THEN** the response carries the usage stage and the state, then the rate-limit error

### Requirement: A failed turn records its error in the state
The state of a turn that fails SHALL hold the raw error as `error`, and the state of a turn that succeeds SHALL hold `error` = `None`. A turn that ends with a DIAL error raised to the client after the turn started (a Deep Research failure, a rate limit, a forbidden deployment) SHALL also save the error's `display_message` as `error_display_message`. Only `error_display_message` is safe to show: the raw `error` can carry internal details such as cost limits.

#### Scenario: Rate limit
- **WHEN** a turn fails with a rate-limit error from the Supreme Agent's model
- **THEN** its state holds the raw error as `error`, and the message shown to the user as `error_display_message`

#### Scenario: Successful turn
- **WHEN** a turn completes without an error
- **THEN** its state holds `error` = `None`

### Requirement: Failures inside a tool call
A failure inside a tool call, including a rate-limit, permission-denied or content-filter error from a model or application the tool calls, SHALL be handed to the agent as an error tool response and SHALL NOT become a DIAL error. The only exception SHALL be a Deep Research deployment failure, which follows `deep-research`, whatever its status or code.

#### Scenario: RAG application without access
- **WHEN** the File RAG call returns 403 for deployment `publications-rag`
- **THEN** the agent gets an error tool response for that call and the turn is answered with status 200

### Requirement: Rate limits
A rate-limit error from a model called outside a tool call SHALL be reported as a DIAL error with HTTP status 500, `code` "429", type `rate_limit_exceeded`, the response header `x-should-retry: false`, and `Retry-After` when the deployment gave one. The status SHALL be 500 because DIAL Core replaces an application's 429 with a bare 503 and drops its body, while a 500 is forwarded intact; `code` keeps the meaning for the client and `x-should-retry` stops OpenAI SDK clients from retrying the 500.

#### Scenario: Status and code
- **WHEN** the Supreme Agent's model returns 429
- **THEN** the response has HTTP status 500, code "429" and the header `x-should-retry: false`

### Requirement: Rate-limit message
The rate-limit `display_message` SHALL name the exceeded limits (minute, daily, weekly, monthly) when the deployment's message reveals them, and say when the limit resets, from `Retry-After` when present. It SHALL NOT tell the user to try again, as that would point at the "Try again" button the Continue note warns against on a turn that carries the note. On a turn that carries the "Continue" button the Continue note follows the message.

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

### Requirement: Rate-limit error details
The rate-limit DIAL error SHALL carry `underlying_deployment`, the deployment id taken from the failed request's URL, and the original error body.

#### Scenario: Agent model hits its limit
- **WHEN** the request to `http://core/openai/deployments/gpt-4.1/chat/completions` returns 429
- **THEN** the error carries `underlying_deployment` = `gpt-4.1` and the deployment's original error body

### Requirement: Continue button after a rate limit
A streamed turn that ends with a rate-limit DIAL error and has something to continue (see "A turn has something to continue") SHALL carry, in its message's `form_schema`, a `button` property with a single submit button titled "Continue". The button SHALL send the user message "Continue".

#### Scenario: Button next to the error
- **WHEN** a streamed turn runs a data query, streams part of an answer and then fails with a rate limit from the Supreme Agent's model
- **THEN** the response carries the partial answer, a `form_schema` with the "Continue" button, and then the rate-limit error

### Requirement: The Continue button doesn't depend on Deep Research
The "Continue" button SHALL NOT depend on Deep Research: where Deep Research is unavailable to the caller, the `form_schema` SHALL hold only the button; where the Deep Research toggle is shown, the same `form_schema` SHALL keep the `deep_research` control next to the button.

#### Scenario: Button and Deep Research toggle together
- **WHEN** a streamed turn on a channel with Deep Research available runs a data query and then fails with a rate limit
- **THEN** its `form_schema` has both the `deep_research` control and the "Continue" button

#### Scenario: Channel without Deep Research
- **WHEN** a streamed turn runs a data query and then fails with a rate limit, on a channel with no `deep_research` tool
- **THEN** its `form_schema` has the "Continue" button and no `deep_research` control

### Requirement: No Continue button elsewhere
A non-streaming turn SHALL NOT carry the "Continue" button, as its response is only the error. A turn that succeeds, or ends with an error other than a rate limit or a Deep Research deployment failure (see `deep-research`), SHALL NOT carry it either. A rate-limited turn with nothing to continue SHALL NOT carry it: "Try again" repeats no tool work there.

#### Scenario: Non-streaming request
- **WHEN** a non-streaming turn fails with a rate limit from the Supreme Agent's model
- **THEN** the response is only the rate-limit error, with no button

#### Scenario: Successful turn
- **WHEN** a streamed turn completes without an error
- **THEN** its `form_schema`, if any, has no `button` property

#### Scenario: Forbidden deployment
- **WHEN** a streamed turn fails because the Supreme Agent's model returns 403
- **THEN** its `form_schema`, if any, has no `button` property

#### Scenario: Rate limit before the agent called a tool
- **WHEN** a streamed turn fails with a rate limit on the out-of-scope check or on the agent's first model call
- **THEN** its `form_schema`, if any, has no `button` property, and the error has no Continue note

#### Scenario: Answer stopped without a tool call
- **WHEN** a streamed turn streams part of an answer without calling a tool, and then fails with a rate limit
- **THEN** the response carries the partial answer and the rate-limit error, with no `button` property and no Continue note

#### Scenario: Out-of-scope refusal stopped
- **WHEN** the out-of-scope check rejects the user's message, and the streamed refusal fails with a rate limit
- **THEN** its `form_schema`, if any, has no `button` property, and the error has no Continue note

#### Scenario: Only configured tool calls ran
- **WHEN** a streamed turn ran only the tool calls the channel configures to run before every turn, and then fails with a rate limit on the agent's first model call
- **THEN** its `form_schema`, if any, has no `button` property

#### Scenario: A "Continue" turn that fails at once
- **WHEN** a turn ran a data query and failed with a rate limit, the user pressed "Continue", and the new turn fails with a rate limit on its first model call
- **THEN** the new turn has no button and no note, while the earlier failed message keeps its state, so "Try again" sends "Continue" again and the agent still sees the saved query

### Requirement: Continue note goes with the button
Every error `display_message` of a turn that carries the "Continue" button SHALL end with the Continue note, after a blank line. The note is this text: `Press "Continue" (or type "continue") to pick up from where the response stopped: what was already done is kept. "Try again" starts over from your last message and repeats that work, so the response takes longer and uses more tokens.`. A turn without the button SHALL NOT get the note.

#### Scenario: Rate limit while streaming
- **WHEN** a streamed turn runs a data query and then fails with a daily rate limit and `Retry-After: 3700`
- **THEN** the user gets "You've exceeded your daily token limit for the underlying deployment. The limit resets in 1 hour 1 minute.", a blank line and the Continue note

#### Scenario: No note without the button
- **WHEN** a non-streaming turn fails with a rate limit
- **THEN** its `display_message` doesn't contain the Continue note

### Requirement: The saved error includes the Continue note
`error_display_message` in the state SHALL hold the full text the user was shown, Continue note included, so the failed-turn notes quote what the user saw.

#### Scenario: Note in the state
- **WHEN** a streamed turn runs a data query and then fails with a rate limit
- **THEN** the state's `error_display_message` ends with the Continue note

### Requirement: A turn has something to continue
A failed turn SHALL have something to continue when it holds a tool call the agent made in this turn, with or without its response: "Try again" would delete it with the failed message. Tool calls the channel configures to run before every turn SHALL NOT count. Content already sent to the user SHALL NOT count. The rule SHALL NOT read the content of a tool response or of Deep Research's state.

#### Scenario: Tool call with an error response
- **WHEN** a streamed turn ran a data query that failed, and the agent's next model call hit a rate limit
- **THEN** the turn has something to continue and carries the "Continue" button

#### Scenario: Content only
- **WHEN** a streamed turn called no tool, streamed part of its answer and then hit a rate limit
- **THEN** the turn has nothing to continue

#### Scenario: Deep Research deployment failure
- **WHEN** a streamed turn ends with a Deep Research deployment failure
- **THEN** the turn has something to continue, as the failure happened inside a Deep Research call, and carries the "Continue" button (see `deep-research`)

#### Scenario: Nothing done
- **WHEN** a streamed turn hit a rate limit before any tool call of the agent
- **THEN** the turn has nothing to continue

### Requirement: A failed turn saves the out-of-scope result
The state of a turn that fails and sends its state SHALL hold the out-of-scope check's result as `out_of_scope` and its reasoning as `out_of_scope_reasoning`, as the state of a successful turn does. A turn that failed before the check ran SHALL hold `None` for both.

#### Scenario: Refusal stopped by a rate limit
- **WHEN** the out-of-scope check rejects the user's message, and the streamed refusal fails with a rate limit
- **THEN** the turn's state holds `out_of_scope` = `true` and the check's reasoning, so the check counts this turn toward the "start a new conversation" threshold on later turns

#### Scenario: Failure before the check
- **WHEN** a turn fails with a rate limit before the out-of-scope check ran
- **THEN** the turn's state holds `out_of_scope` = `None` and `out_of_scope_reasoning` = `None`

### Requirement: A rate-limited turn saves its tool messages
A turn that fails with a rate-limit DIAL error SHALL save in its state the tool calls and responses it completed before the error, so a later turn can reuse them instead of calling the tools again.

#### Scenario: Rate limit after a data query
- **WHEN** a turn ran a data query tool and the Supreme Agent's model then hit a rate limit while writing the answer
- **THEN** the turn's state holds the tool call and its response, the failed message carries the query's data attachments (see `data-query-agent-response`), and on the next turn ("Continue") the agent sees the saved call and does not need to run the query again

#### Scenario: Rate limit after Deep Research replied
- **WHEN** in a Deep Research turn, Deep Research returned a clarifying question and the Supreme Agent's model then hit a rate limit before the question was shown
- **THEN** the turn's state holds the Deep Research call and its response next to the session, so on the next turn ("Continue") the agent shows the pending question instead of passing "Continue" on to Deep Research as the user's answer

### Requirement: Forbidden deployments
A permission-denied error from a model or application the turn calls outside a tool call SHALL be reported as a DIAL error with status 403, `code` "403", type `forbidden_deployment` and a `display_message` saying the user has no access to the underlying deployment, naming it when known, and to contact the administrator. The error SHALL carry `underlying_deployment`, the deployment id read from the failed request's URL, and the original error body.

#### Scenario: Agent model without access
- **WHEN** the Supreme Agent's model returns 403 for deployment `gpt-4o`
- **THEN** the user gets a 403 error saying "You don't have access to the underlying deployment 'gpt-4o'. Please contact your administrator."

### Requirement: Content filter
A request refused by the content filter of a model called outside a tool call SHALL be reported as a DIAL error with status 400 and code `content_filter`, carrying the model's error fields, and so SHALL a model response that stops for a content-filter reason.

#### Scenario: Filtered prompt
- **WHEN** the Supreme Agent's model rejects the prompt with code `content_filter`
- **THEN** the response is a 400 DIAL error with code `content_filter`

### Requirement: Invalid message history
A request whose messages cannot form a history SHALL be answered with status 200 and the text "This conversation contains a message that cannot be processed. Please start a new chat.". A history SHALL be invalid when a user message has empty content, a tool message has no `tool_call_id`, an assistant message carries an invalid tool call, or the saved tool messages have an unknown type.

#### Scenario: Empty user message
- **WHEN** the second user message of a request has empty content
- **THEN** the response is a 200 with the "start a new chat" text

### Requirement: DIAL errors pass through
A DIAL error raised during the turn (e.g. a request rejected by validation) SHALL be delivered unchanged, not replaced by a 200 message.

#### Scenario: System message with content
- **WHEN** a request carries a system message with non-empty content
- **THEN** the response is the invalid-request DIAL error saying such system messages are not supported

### Requirement: Invalid history is diagnosable
An invalid history SHALL be logged as one warning without a traceback, and the received messages SHALL be saved in the state as `received_messages`, without the `custom_content` of assistant messages and without attachment data, which would dwarf the rest. The reply SHALL be a 200 rather than a DIAL error because the SDK discards the accumulated response, including that dump, when a DIAL error reaches the stream.

#### Scenario: Dump kept
- **WHEN** a request is answered with the "start a new chat" text
- **THEN** the response state holds `received_messages`

### Requirement: Message with only commands
A user message that consists only of chat commands (e.g. `!show_debug_stages`) SHALL be answered with status 200 and the text "Your message contains only commands. Please add your question and send it again.", without a message dump.

#### Scenario: Command alone
- **WHEN** the latest user message is only a command
- **THEN** the reply asks the user to add a question and resend

### Requirement: Other failures
Any other failure of the turn, including other bad-request errors from a deployment, SHALL be logged and answered with status 200 and "An error occurred while processing your request." appended to the answer, and the error SHALL be saved in the state. A failure to build the caller's auth context that is not itself a DIAL error SHALL be refused with an internal server error.

#### Scenario: Unexpected exception
- **WHEN** a tool chain raises an unexpected exception that the agent loop does not handle
- **THEN** the answer ends with "An error occurred while processing your request." and the state records the error

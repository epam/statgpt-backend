# chat-error-handling Specification

## Purpose

Defines how a chat completion request to a channel reports failures: which ones become DIAL errors with which status, code and user-facing `display_message`, which ones are answered with a plain message instead, and what the response still carries when a turn fails. Deep Research's own failure rules belong to `deep-research`.

## Requirements

### Requirement: Unknown channel
A chat completion or configuration request for a deployment id with no channel SHALL be refused with status 404, code `deployment_not_found` and the message "The API deployment for this resource does not exist.".

#### Scenario: Missing channel
- **WHEN** a request arrives for deployment `missing` and no channel has that deployment id
- **THEN** the response is a 404 DIAL error with code `deployment_not_found`

### Requirement: Usage and state survive a DIAL error
When a turn fails with an error that is reported as a DIAL error after the turn started (Deep Research failure, rate limit, forbidden deployment), the response SHALL first emit the token usage stage and the conversation state, and only then raise the error, so the usage is accounted and the next turn keeps its state.

#### Scenario: Rate limit mid-turn
- **WHEN** a nested model call hits a rate limit after other model calls of the turn succeeded
- **THEN** the response carries the usage stage and the state, then the rate-limit error

### Requirement: Rate limits
A rate-limit error from a deployment the turn calls SHALL be reported as a DIAL error with HTTP status 500, `code` "429", type `rate_limit_exceeded`, the response header `x-should-retry: false`, and `Retry-After` when the deployment gave one. The status SHALL be 500 because DIAL Core replaces an application's 429 with a bare 503 and drops its body, while a 500 is forwarded intact; `code` keeps the meaning for the client and `x-should-retry` stops OpenAI SDK clients from retrying the 500.

#### Scenario: Status and code
- **WHEN** a nested model returns 429
- **THEN** the response has HTTP status 500, code "429" and the header `x-should-retry: false`

### Requirement: Rate-limit message
The rate-limit `display_message` SHALL name the exceeded limits (minute, daily, weekly, monthly) when the deployment's message reveals them, and say when to retry, from `Retry-After` when present. The error SHALL carry `underlying_deployment`, the deployment id taken from the failed request's URL, and the original error body.

#### Scenario: Daily limit
- **WHEN** a nested model returns 429 with "Day limit: 120000 / 100000 tokens" and `Retry-After: 3700`
- **THEN** the user gets "You've exceeded your daily token limit for the underlying deployment. Please try again in 1 hour 1 minute", with HTTP status 500 and code "429"

### Requirement: Forbidden deployments
A permission-denied error from a deployment the turn calls SHALL be reported as a DIAL error with status 403, `code` "403", type `forbidden_deployment` and a `display_message` saying the user has no access to the underlying deployment, naming it when known, and to contact the administrator. The error SHALL carry `underlying_deployment` and the original error body. Because the deployment id is read from the request URL, this SHALL cover DIAL applications as well as models.

#### Scenario: RAG application without access
- **WHEN** the File RAG call returns 403 for deployment `publications-rag`
- **THEN** the user gets a 403 error saying "You don't have access to the underlying deployment 'publications-rag'. Please contact your administrator."

### Requirement: Content filter
A request refused by a model's content filter SHALL be reported as a DIAL error with status 400 and code `content_filter`, carrying the model's error fields, and so SHALL a model response that stops for a content-filter reason.

#### Scenario: Filtered prompt
- **WHEN** a nested model rejects the prompt with code `content_filter`
- **THEN** the response is a 400 DIAL error with code `content_filter`

### Requirement: Invalid message history
A request whose messages cannot form a history SHALL be answered with status 200 and the text "This conversation contains a message that cannot be processed. Please start a new chat.". A history SHALL be invalid when a user message has empty content, a tool message has no `tool_call_id`, an assistant message carries an invalid tool call, or the saved tool messages have an unknown type. DIAL errors raised during the turn (e.g. a request rejected by validation) SHALL pass through unchanged.

#### Scenario: Empty user message
- **WHEN** the second user message of a request has empty content
- **THEN** the response is a 200 with the "start a new chat" text

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

# deep-research Specification

## Purpose

Lets a chat user run an in-depth investigation by the separate Deep Research DIAL application from a StatGPT channel. The user turns the mode on with a per-message toggle; the Supreme Agent mediates Deep Research's clarifying questions and plan approval, and Deep Research's final report is delivered to the user verbatim. How the stages, attachments and annotations of the Deep Research response are relayed belongs to `sub-deployment-relay`; how other failures reach the user belongs to `chat-error-handling`.

## Requirements

### Requirement: Deep Research is reachable only through the toggle
Deep Research SHALL be reachable only through the `deep_research` field of the chat request's configuration (the "Deep research" toggle). It SHALL NOT be part of the channel's tool list, so it SHALL never be offered to the general Supreme Agent, exposed over MCP, or described to the out-of-scope checker. Whether a turn is a Deep Research turn SHALL be decided only from the toggle and the presence of an in-progress session, never by interpreting the message text.

#### Scenario: Research question without the toggle
- **WHEN** a user asks "Analyse the drivers of inflation in Europe since 2020" with the toggle off
- **THEN** the turn is handled by the general Supreme Agent, which has no Deep Research tool

### Requirement: Availability per channel
Deep Research SHALL be available on a channel only when its `deep_research` tool is configured and enabled. Where Deep Research is unavailable to the caller, the deployment configuration SHALL NOT advertise the toggle, and a request that sets `deep_research: true` anyway SHALL be handled as a normal turn. The toggle's configuration default SHALL be false.

#### Scenario: Tool disabled
- **WHEN** the channel's `deep_research` tool has `enabled: false`
- **THEN** the configuration endpoint lists no "Deep research" toggle

### Requirement: Availability per user
When the tool sets `accessClaimValue`, Deep Research SHALL be available only to callers whose DIAL roles, resolved from their access token through DIAL's user-info endpoint, include that value. If a role is required but the roles cannot be resolved, access SHALL be denied. System users SHALL always have access, so evaluations can run without a user token; they are disabled in production.

#### Scenario: User without the required role forces the toggle
- **WHEN** the tool has `accessClaimValue: deep-research-users` and a user without that role sends a request with `deep_research: true`
- **THEN** the configuration endpoint does not list the toggle for that user and the request is answered as a normal turn

#### Scenario: Roles cannot be resolved
- **WHEN** a role is required and the user-info call fails
- **THEN** Deep Research is unavailable to that caller

### Requirement: Routing of a turn
With Deep Research available to the caller, a turn SHALL be routed as follows:
- toggle on and no session in progress: start a new session;
- toggle on and a session in progress: resume the session;
- toggle off and a session in progress: drop the session and handle the turn normally;
- toggle off and no session: a normal turn.

A session SHALL count as in progress exactly while it is stored in the conversation state.

#### Scenario: Toggle turned off mid-session
- **WHEN** Deep Research asked a clarifying question on the previous turn and the user now sends a message with the toggle off
- **THEN** the session is dropped from the state and the message is answered as a normal turn

### Requirement: Abandoned sessions do not come back
A session dropped because the toggle was off SHALL be gone for good, so the abandoned run neither resumes silently nor comes back when the user turns the toggle on again later in the chat.

#### Scenario: Toggle turned on again later
- **WHEN** the user turned the toggle off mid-session and later turns it on again in the same chat
- **THEN** a new session is started rather than the abandoned one resumed

### Requirement: Query check before a new session
Before a new session starts, an LLM check (`queryCheck`) SHALL decide from the conversation whether the latest message suits Deep Research, so that a message that is not a research question does not launch a slow, expensive run. Topics listed in `excludedTopics` SHALL NOT suit Deep Research; when it is null, defaults apply (the assistant's capabilities, navigation help, term definitions, small talk). With `enabled: false` every message sent with the toggle on SHALL start a session.

#### Scenario: Query does not suit Deep Research
- **WHEN** the toggle is on, no session is in progress, and the user asks "What can you do?"
- **THEN** no session is started, `deactivationMessageToUser` is shown before the answer (nothing if it is empty), the message is answered as a normal turn, and the toggle is disarmed for the next request

### Requirement: When the query check runs
The query check SHALL run only on a turn that would start a session, never on messages sent while a session is in progress. It SHALL run concurrently with the turn's fake-tool-call prelude, so it delays the turn only by the time it outlasts the prelude.

#### Scenario: Answer to a clarifying question
- **WHEN** a session is in progress and the user answers "Only euro area countries"
- **THEN** the query check does not run

### Requirement: Query check fails open
The query check only spares an unnecessary run, so it SHALL fail open: if it fails or does not finish within `timeoutSeconds` (default 10, including the model client's retries and their backoff), Deep Research SHALL be started.

#### Scenario: Check times out
- **WHEN** the check model does not answer within `timeoutSeconds`
- **THEN** Deep Research is started

### Requirement: Agent is told about the mode
On a turn that would start a session, the agent SHALL get a system message saying that the user enabled Deep Research mode, and, when the query check rejected the query, a second one saying that the mode was disabled and the request should be answered as a regular one. These messages SHALL be saved with the turn's tool messages, so later turns see them.

#### Scenario: Rejected query on a later turn
- **WHEN** the query check rejected a query and the user asks a follow-up with the toggle off
- **THEN** the agent's history holds both system messages of the earlier turn

### Requirement: Deep Research turns run in their own loop
A Deep Research turn SHALL run in its own loop, where the Supreme Agent has only the Deep Research tools, no parallel tool calls, and the Deep Research prompt section. On a starting turn the first call SHALL be forced to the start tool (`deep_research`); later, the agent SHALL choose between the resume tool (`resumeToolName`, default `resume_deep_research`) and a message to the user. The fake-tool-call prelude SHALL still run, so its results can help answer clarifying questions.

#### Scenario: Starting turn
- **WHEN** a user sends a research question with the toggle on and the query check accepts it
- **THEN** the agent's first call is to the start tool with the question

### Requirement: Supreme Agent mediates clarifications and the plan
A clarification or plan SHALL be shown in the tool-result stage and handed back to the agent, which SHALL answer the clarifying questions it can from the conversation context through the resume tool, surface to the user only the questions it cannot answer, and present a plan verbatim, approving it only once the user does. When the user asks for something unrelated during a session, the agent SHALL ask them to answer the pending question or to turn the mode off.

#### Scenario: Questions partly answered from context
- **WHEN** Deep Research asks for the countries and the period, and the user already named the countries earlier in the chat
- **THEN** the agent sends the countries through the resume tool and asks the user only for the period

#### Scenario: Plan for approval
- **WHEN** Deep Research presents a research plan
- **THEN** the agent shows it to the user verbatim and the turn ends without approving it

### Requirement: Mediation loop ends
The mediation loop SHALL end when the agent writes to the user or Deep Research delivers its final report. It SHALL also stop after 50 iterations, set far above any realistic number of clarification rounds, as a guard against a deployment that never finishes or an agent stuck producing invalid responses. On reaching the cap, the standard failure text SHALL be appended to the answer once and the session SHALL be kept, so the next turn with the toggle on resumes it.

#### Scenario: Cap reached
- **WHEN** 50 mediation iterations pass without the agent writing to the user or a report being delivered
- **THEN** the answer ends with "Deep Research could not complete this request. Please try again." and the session stays in the state

### Requirement: Session persistence
The sub-conversation with Deep Research SHALL be stored in the conversation state as a list of turns (`user_message`, `assistant_content`, `deep_research_state`) and carried over from the latest assistant message that has a state. It SHALL be replayed to Deep Research as DIAL messages with each assistant turn's `custom_content.state`, so Deep Research resumes from its own state. A session SHALL be saved once Deep Research answers and dropped once the report is delivered.

#### Scenario: Clarification round trip
- **WHEN** Deep Research answers the first question with a clarifying question
- **THEN** the next request replays the original question, Deep Research's reply with its state, and the new message

### Requirement: Session turns are not stored as messages
The stored turns SHALL use neutral field names rather than the DIAL message shape (`role`, `content`, `custom_content`), because the DIAL chat client strips `custom_content` from message-shaped objects in the state, which would drop Deep Research's state and force a re-plan every turn.

#### Scenario: State round trip through the chat client
- **WHEN** the chat client sends the state of a clarification turn back on the next request
- **THEN** Deep Research's state of each turn is still present

### Requirement: Final-report contract
StatGPT SHALL treat the investigation as finished, and the response as the final report, exactly when Deep Research's state carries `preparation.research_started: true`. This relies on Deep Research setting that flag only after the plan is approved and the final report has fully streamed in the same response; if it were set earlier, the session would be closed before the report arrived.

#### Scenario: Flag set
- **WHEN** a Deep Research response ends with a state where `preparation.research_started` is true
- **THEN** its content is delivered as the final report

### Requirement: Content is routed when the response ends
Because a clarification and the final report cannot be told apart until the response ends, the content of a Deep Research response SHALL be buffered and routed once the stream finishes, while its progress stages stream live.

#### Scenario: Long report
- **WHEN** Deep Research streams a report for several minutes
- **THEN** its progress stages appear as they arrive and the report text appears once the stream ends

### Requirement: Final report is delivered verbatim
The final report SHALL be appended to the user's answer verbatim, content and attachments (e.g. a Canvas document) as Deep Research sent them, without the agent repeating, summarizing or paraphrasing it. The turn SHALL then end, the session SHALL be dropped, and the agent SHALL get the system message `reportDeliveredMessageToAgent` (or the default saying the mode was disabled because the report was delivered), saved with the turn.

#### Scenario: Report delivered
- **WHEN** Deep Research delivers its report
- **THEN** the user sees the report exactly as sent, the agent writes nothing after it, and the state holds no session

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

### Requirement: Report and its sources stay in the agent's history
The report SHALL stay in the agent's history as the tool response of the call that delivered it, followed by a citations note listing, for each distinct citation tag of the report, its id, the title of the cited source and the quoted passage (each left out when the annotation has none), so the agent can answer follow-up questions about the sources. The user SHALL NOT see the note.

#### Scenario: Follow-up about a source
- **WHEN** the user asks "Where does the 2.4% figure come from?" after a delivered report
- **THEN** the agent's history holds the report as a tool response with the citations note naming the tag's source

### Requirement: Report is not repeated in the agent's history
On later turns the report's assistant message SHALL be replaced in the agent's history by a short placeholder saying the report was delivered verbatim. The tool response is the copy kept because the agent treats a tool response as a source, but would re-check its own earlier answer, e.g. by searching again for the source of a figure in the report.

#### Scenario: Turn after the report
- **WHEN** the user sends a message after a turn that delivered a report
- **THEN** the agent's history holds the report once, as the tool response, and a placeholder in place of the report message

### Requirement: Resume stage names
The resume tool's stage names SHALL come from `resumeStagesConfig` (`toolCallName`, default "Researching: {original_question}"; `toolResultName`, default "Research result: {original_question}"), never from the start tool's stage names, because the resume tool takes a `message` argument instead of `query`. They SHALL support the `{original_question}` placeholder, the question the session was started with, and `{message}`.

#### Scenario: Resume call
- **WHEN** a session started with "Drivers of inflation in Europe" is resumed
- **THEN** the stage is named "Researching: Drivers of inflation in Europe"

### Requirement: Missing original question
When the original question cannot be read from the session, a resume stage name that references `{original_question}` SHALL be replaced entirely by `missingQuestionName` (default "Continue researching") rather than shown half-filled.

#### Scenario: No session in the state
- **WHEN** the resume tool is called and no session turn is stored
- **THEN** its stage is named "Continue researching"

### Requirement: Caller's token is forwarded
Requests to Deep Research SHALL carry the caller's `api-key` and, when the caller has an access token, `Authorization: Bearer <token>`, because Deep Research queries data back through this application's MCP server and needs the user's token to authorize downstream data sources. Whether DIAL Core relays the token SHALL be governed only by its `forwardAuthToken` setting.

#### Scenario: User with a token
- **WHEN** a user signed in with a bearer token starts Deep Research
- **THEN** the request to the Deep Research deployment carries both the api key and that bearer token

### Requirement: Requests are streamed
Requests to Deep Research SHALL be streamed, because in a blocking response the DIAL SDK merges the chunks and strips the `index` of every annotation, and the report's citations would silently break.

#### Scenario: Report with citations
- **WHEN** Deep Research delivers a report with citation annotations
- **THEN** the annotations reach the user with their indexes

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

### Requirement: Failed turns can be resumed
If Deep Research sent a state before failing, the failed turn SHALL be kept in the session with that state, so Deep Research's next call can decide whether to resume; otherwise the session SHALL be left as it was. The failed call SHALL be answered in the agent's history with an error tool response and, when a session is kept, a system note telling the agent to call the resume tool if the user asks to continue or retry. The history SHALL be saved with the failed turn.

#### Scenario: User asks to continue after a failure
- **WHEN** a failed turn left a session and the user writes "continue" with the toggle on
- **THEN** the agent resumes the session through the resume tool

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

### Requirement: Other failures append the standard text
A failure of a Deep Research tool call other than a deployment error SHALL be recorded as an error tool response, and the turn SHALL end with the standard failure text, "Deep Research could not complete this request. Please try again.", appended to the answer once. Unlike the text of a deployment failure, it SHALL tell the user to try again: the turn raises no DIAL error, so it has neither the Continue note nor the "Try again" button.

#### Scenario: Unexpected exception
- **WHEN** the tool raises an exception that is not a deployment error
- **THEN** the answer ends with the standard failure text and no DIAL error is raised

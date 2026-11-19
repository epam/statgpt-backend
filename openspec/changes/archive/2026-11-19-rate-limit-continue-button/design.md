# Design

## Context

- **Error flow in `ChannelCompletion._channel_completion`.**
  - It turns `openai.RateLimitError` into `RateLimitException.from_openai_error(e)`, and `DeepResearchFailedError` into a generic 500 `DIALException` with Deep Research's `display_message`.
  - Either way, it then emits the usage stage, the Deep Research form schema and the state, and only then raises the error.
  - In a streamed response, everything sent before the error reaches the client. In a non-streaming response the SDK discards it and returns only the error.
  - `request.stream` says which kind of response it is; it defaults to `false`. DIAL Chat always streams.
- **The SDK allows one form schema per choice.** `Choice.set_form_schema` raises "Trying to set form schema twice" on a second call. Today the only form schema is the Deep Research toggle from `build_deep_research_form_schema`, sent only on channels where Deep Research is available.
- **DIAL Chat** (checked on `development` at 497040770). The StatGPT Portal Frontend renders the button too (checked by the user).
  - Buttons come only from a `button` or `starter` property (`getStartersFromSchema`).
  - Toggles come from the boolean properties of the same schema (`getToolConfigurationFromFormSchema`), so a string-valued `button` doesn't disturb `deep_research`.
  - A submit button sends its `populateText` as the user message, with `form_value: {button: <const>}`, and keeps the active toggles in `configuration_value`.
  - The error box's "Try again" regenerates from the last user message.
- **Data attachments are added only with the final answer.** `SupremeAgentExecutor.stream_response` calls `data_displayer.display` right before it returns the answer, with the artifacts of this turn's Data Query calls. The state keeps only an artifact's `state`, not its data, so a later turn can't rebuild the attachments of an earlier turn's query.
- **Only the Deep Research failure branch saves tool messages** (`History.dump_state`). A rate-limited turn's state has no `tool_messages`.
- **Deep Research sessions.** A session lives in the state as `deep_research_session` and is copied forward from the latest assistant message that has a state.
  - A turn with the toggle on and a session in progress resumes it, through a mediation loop that calls the Supreme Agent's model.
  - If that model hits a rate limit after Deep Research replied, the session already holds Deep Research's new turn, but the tool exchange is lost. The agent then can't see the question it was about to show.
- **The agent's notes already handle "continue".** `failed-turn-history` and `deep-research` (`deepResearchFailedMessageToAgent`) tell the agent what to do when the user says it.

## Goals / Non-Goals

**Goals:**
- One form schema per message that can hold both the Deep Research toggle and the "Continue" button.
- One place that decides on the button and adds the note, so the two always go together.
- "Continue" reuses the finished tool calls, including the Deep Research exchange of a rate-limited turn.
- The user still gets the tables and graphs of the queries that finished before the error.

**Non-Goals:**
- A button for other errors: forbidden deployment, content filter, or a Deep Research tool failure that isn't a deployment error.
- A button when the agent loop reaches its iteration cap. That turn isn't an error, and its answer already asks the user to enter "continue".
- Backend handling of `form_value`. The turn after the click is a normal turn whose message is "Continue", and a typed "continue" works the same way.
- Choosing which turn "Continue" finishes after several failed turns in a row, such as an interrupted answer followed by a new question that also failed. The agent judges from the history and the existing notes; no new rule or note text.
- A reminder about an unfinished answer when the user moves on with a new question.
- Hiding or relabeling DIAL Chat's "Try again" button; that is the frontend's domain.
- Localizing the new texts; the current rate-limit message isn't localized either.

## Decisions

### One form-schema builder for the message
Replace the direct `build_deep_research_form_schema` call with a builder that takes the optional toggle value and whether to add the button. It returns `None` when the schema would be empty, and `ChannelCompletion` calls `set_form_schema` once. Where Deep Research is unavailable to the caller, the toggle value is absent and a continuable turn gets a button-only schema. `build_deep_research_form_schema` either stays as the toggle-only case or is folded into the new builder; `FormMetaclass` builds the class dynamically either way.
- Alternative: set the schema in two places. Rejected because the SDK raises on a second call.
- Alternative: merge two JSON schemas by hand. Rejected: it duplicates what `FormMetaclass` produces and breaks quietly when the SDK's schema shape changes.

### Button shape
- Property `button`, `str` type, one `Button(const="continue", title="Continue", submit=True, populateText="Continue")`, no `description`.
- `button` is one of the two keys DIAL Chat renders.
- `populateText` makes the user bubble and the request read "Continue", which the existing notes already handle.
- No description, because the error box right above it already holds the note.

### One decision point for the button and the note
`ChannelCompletion` decides "continuable" once: the request is streamed, and the turn ended with a rate limit or a Deep Research deployment failure, whether or not a session was kept. When it is:
- it appends the note to the DIAL error's `display_message`, after a blank line;
- it adds the button to the form schema;
- the full text goes into `error_display_message`, since it is what the user saw.

The note is one module constant, with the exact text from the `chat-error-handling` delta.
- The note starts with `Press "Continue" (or type "continue")`, so it stays true where a client renders no button.
- Alternative: build the note into each error's own message. Rejected: Deep Research's text comes from another service, and the rate-limit message would need to know about streaming and the button.
- Alternative: store the message without the note in the state. Rejected: the failed-turn notes quote what the user saw, and the note does no harm to the agent; if anything, it helps it read a "Continue".

### Rate-limit message without "try again"
`RateLimitException` builds only the core message, the same for every request:
- the exceeded limits;
- then "The limit resets in <duration>.", or "Please wait for the limit to reset." without `Retry-After`.

It needs no flag for streaming. The non-streaming text changes slightly, which is acceptable: status, code and headers stay as they are.

### Deep Research fallback text without "Please try again."
`_DEEP_RESEARCH_ERROR_TEXT` becomes "Deep Research could not complete this request.". It is one constant for the deployment failure without a `display_message`, the mediation cap and other tool failures, and it stays one.
- Alternative: a second text only for the deployment failure. Rejected: two near-identical texts, and in the other two cases the sentence adds little, since the answer ends there and the user can simply write again.

### Data attachments whenever the agent loop ends
`stream_response` adds the data attachments of the collected artifacts on every way out of the agent loop: the final answer, the iteration cap and an exception.
- On an exception, the attachments are added and the exception is raised again. A failure while adding them is logged and dropped, so the user gets the original error.
- Only `Exception` is handled, not cancellation: a turn the client cancelled has nobody to show attachments to.
- The failed message then carries the attachments, and the continued answer right below it refers to them.
- Alternative: keep the data in the state and add the attachments on the "Continue" turn. Rejected: it puts full datasets into the state of every message.
- Alternative: have "Continue" rerun the queries. Rejected: it is the cost this change removes.

### Save tool messages on a rate limit
In the `openai.RateLimitError` branch, call `ChainParameters.get_history(inputs).dump_state(state)`, as the Deep Research branch does.
- If the history was never built, `inputs` has no `HISTORY` and there is nothing to save, so the call needs a guard.
- A tool's own rate limit becomes an error tool response and never reaches this branch. That leaves only completed tool-call/response pairs in the history.
- No dangling call is left: a tool call is added only after the model's response is complete. This includes the Deep Research mediation loop, which adds its exchange to the same history.

## Risks / Trade-offs

- [Users press "Continue" before the limit resets] → The turn fails again with the same note and button. The earlier turn's tool messages stay in that turn's state, so a later "Continue" still sees them; only the wasted call costs anything.
- [Deep Research's own text says "Please retry" right before a note that warns against "Try again"] → The note says which button to use, so the two read as one instruction. We don't control Deep Research's text.
- ["Continue" after a Deep Research failure with no session starts a new run] → The failed call and its error are in the history, so the agent knows what happened. The note's "what was already done is kept" covers only that history in this case.
- [Attachments split across two messages when "Continue" runs more queries] → Each message carries the attachments of its own queries; nothing is lost or shown twice.
- [Attachments now also appear on turns that fail in other ways, e.g. with "An error occurred while processing your request."] → The data was fetched and the result stages already show the queries, so it is an improvement rather than a risk.
- [Larger state on rate-limited turns] → It's the same size as a successful turn's state.
- [Rate-limited turns from before the deploy have no tool messages] → "Continue" reruns their tools, as today.

## Migration Plan

No data migration. Deploy the chat backend; roll back by reverting. The response's status, code and headers don't change.

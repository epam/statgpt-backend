# Proposal

## Why

When a streamed turn hits a rate limit, the only way on is the error box's "Try again" button. That regenerates the response from the last user message, so the agent redoes the tool calls it already finished. The response takes longer and uses more tokens, which matters even more because the user has just run out of tokens. Sending "continue" instead lets the agent resume from where it stopped (`failed-turn-history`, `deep-research`), but users don't know that.

There is also a gap in the state. A rate-limited turn doesn't save its tool calls, so even "continue" repeats them. In Deep Research it does worse: if the limit hits after Deep Research replied, the session holds a clarifying question the agent never sees. "Continue" is then passed on to Deep Research as if it were the user's answer.

## What Changes

- **Continue button.** A streamed turn that ends with a rate-limit DIAL error or a Deep Research deployment failure carries a "Continue" button next to the error.
  - The button is a `button` property in the message's `form_schema`, in the same schema as the Deep Research toggle.
  - Clicking it sends the user message "Continue".
  - It doesn't depend on Deep Research: on a channel without it, the schema holds only the button.
  - After a Deep Research failure, the button appears whether or not a session was kept: the failed call and its error stay in the history and help the next turn.
- **Continue note.** Whenever the button is shown, the error's `display_message` ends with one shared note.
  - The note says to press "Continue" (or type "continue") to pick up where the response stopped, because what was already done is kept.
  - It warns that "Try again" starts over from the last user message and repeats that work, so it takes longer and uses more tokens.
  - The note follows whatever the error said, including Deep Research's own text.
- **Rate-limit message.** It no longer tells the user to "try again", which would point at the button the note warns against. It still says which limit was exceeded, and now says when the limit resets. This applies to non-streaming requests too.
- **Deep Research fallback message.** "Deep Research could not complete this request. Please try again." loses its last sentence, for the same reason. The one text is used for a deployment failure without a `display_message`, the mediation cap and other tool failures, so all three change.
- **Data attachments on a failed turn.** The data attachments of the queries that finished are added even when the turn then fails or reaches the iteration cap, not only with the final answer. Otherwise a "Continue" that answers from the saved tool calls would show no tables or graphs.
- **Failed-turn notes.** The notes that quote an error tell the agent the error may be resolved "even if it said to try again later". They now say "to wait or to try again later", as the new rate-limit message tells the user to wait for the limit to reset.
- **Saved tool calls.** A rate-limited turn saves its tool calls and responses in the state, as a failed Deep Research turn already does. "Continue" can then reuse them, and in Deep Research the agent sees the reply Deep Research gave before the limit.
- **Non-streaming requests** get neither the button nor the note. Their response is only the error, so nothing is saved and there is nothing to continue.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `chat-error-handling`:
  - The rate-limit message changes.
  - New: the "Continue" button on rate-limited streamed turns.
  - New: the Continue note that goes with the button.
  - New: a rate-limited turn saves its tool messages.
- `deep-research`:
  - A deployment failure's text gets the Continue note on streamed turns.
  - The message's `form_schema` can hold the button without the `deep_research` control.
  - The standard failure text no longer says "Please try again.".
  - New: the "Continue" button after a deployment failure.
- `data-query-agent-response`: data attachments are added when the turn ends, including a failed turn and the iteration cap.
- `failed-turn-history`:
  - The tool messages a failed turn saves now include a rate-limited turn's, not only a Deep Research failure's.
  - The scenarios quote the new rate-limit message.
  - The notes that quote an error also cover an error that said to wait.

## Impact

- Code:
  - `statgpt/app/utils/dial_exceptions.py`: `RateLimitException` display message.
  - `statgpt/app/application/channel_completion.py`: failure branches, form schema emission, Continue note.
  - `statgpt/app/services/chat_facade.py`: form schema builder.
  - `statgpt/app/chains/deep_research/errors.py`: fallback text.
  - `statgpt/app/default_prompts/assets/supreme_agent.yaml`: failed-turn notes.
  - `statgpt/app/chains/supreme_agent.py`: data attachments when the agent loop fails or reaches the cap.
- Tests:
  - `tests/unit/app/utils/test_dial_exceptions.py`
  - `tests/unit/app/application/test_channel_completion_state.py`
  - `tests/unit/app/application/test_deep_research_toggle_form_schema.py`
  - `tests/unit/app/utils/test_message_history.py`
  - `tests/unit/app/chains/test_deep_research_routing.py`
  - a test of the agent loop's attachments on failure
- Clients:
  - DIAL Chat and StatGPT Portal Frontend render the button from a `button` property; both were checked.
  - MCP clients aren't affected: MCP tool calls don't go through this path.
- API: the response's error code, status and headers don't change; only `display_message` and `form_schema` do.

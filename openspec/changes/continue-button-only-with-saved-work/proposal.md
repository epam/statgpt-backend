# Proposal

## Why

The "Continue" button and its note now appear on every streamed turn that ends with a rate limit. The note says "what was already done is kept" and warns against "Try again". When the turn failed before it did anything, for example at the guardrail step or on the agent's first model call, nothing was done. The note is then wrong, and "Try again" is the natural action: it resends the same message and loses nothing.

A turn stopped in the middle of its answer has the opposite problem. The button is shown, but the agent is told to give the full response again on "Continue", so the button saves nothing there.

## What Changes

- **The button needs something to continue.** A streamed turn that ends with a rate limit or a Deep Research deployment failure carries the "Continue" button and the Continue note only when it holds something that "Try again" would delete:
  - a tool call the agent made in this turn, with or without its response. Configured ("fake") tool calls that run before every turn don't count;
  - or content already sent to the user in this turn.
- **Otherwise the turn ends with the plain error.** It has no button and no note, and the user presses "Try again".
  - The rate-limit message stays as it is: it names the limit and says when it resets.
  - If the failed turn answered a "Continue", "Try again" sends "Continue" again. The work an earlier failed turn saved is still there.
- **Deep Research deployment failures keep the button.** Such a failure always happens inside a Deep Research call, so the turn always holds a tool call. No text or behavior changes there.
- **"Continue" picks up a stopped answer.** The note the agent gets about a turn stopped mid-answer changes:
  - on "continue", the agent goes on from where the response stopped and does not repeat the part the user already saw;
  - on "retry", it gives the full response, as today.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `chat-error-handling`:
  - The "Continue" button after a rate limit needs a tool call or content in the turn.
  - A rate-limited turn with neither gets no button and no note.
  - The rate-limit message is followed by the Continue note only on a turn with the button.
- `failed-turn-history`: the note after a turn stopped mid-answer tells the agent to continue from where the response stopped when the user asks to continue.

## Impact

- Code:
  - `statgpt/app/application/channel_completion.py`: the decision on the button and the note.
  - `statgpt/app/default_prompts/assets/supreme_agent.yaml`: `interruptedTurnMessageToAgent`.
- Tests:
  - `tests/unit/app/application/test_channel_completion_state.py`
  - `tests/unit/app/utils/test_message_history.py`
- Clients: no format change. DIAL Chat and StatGPT Portal Frontend already handle a failed message with and without the button.
- API: the error's status, code and headers don't change. Only the presence of the button and the note does.
- Branch: this goes into `feat/831-rate-limit-continue-button`, before the feature is released, so no user sees the earlier behavior.

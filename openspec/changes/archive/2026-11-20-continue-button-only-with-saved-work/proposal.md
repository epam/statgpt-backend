# Proposal

## Why

The "Continue" button and its note now appear on every streamed turn that ends with a rate limit. The note says "what was already done is kept" and warns against "Try again". When the agent called no tool before the failure, nothing worth keeping was done: the turn failed at the guardrail step, on the agent's first model call, or while writing an answer without tools. The note is then wrong, and "Try again" is the natural action: it resends the same message and repeats no tool work.

A turn stopped mid-answer after tool calls has another problem. The button is shown, but on "Continue" the agent is told to give the full response again, so the part already shown is repeated.

A failed turn also doesn't save the out-of-scope check's result in its state. So the check doesn't count a refusal stopped by an error on later turns.

## What Changes

- **The button needs an agent tool call.** A streamed turn that ends with a rate limit or a Deep Research deployment failure carries the "Continue" button and the Continue note only when the agent made a tool call in this turn, with or without its response. Configured ("fake") tool calls that run before every turn don't count. Content sent to the user doesn't count either.
- **Otherwise the turn ends with the plain error.** It has no button and no note, and the user presses "Try again".
  - The rate-limit message stays as it is: it names the limit and says when it resets.
  - A partial answer written without tool calls is written again on "Try again".
  - An out-of-scope refusal never has an agent tool call, so it never gets the button.
  - If the failed turn answered a "Continue", "Try again" sends "Continue" again. The work an earlier failed turn saved is still there.
- **Deep Research deployment failures keep the button.** Such a failure always happens inside a Deep Research call, so the turn always holds a tool call. No text or behavior changes there.
- **"Continue" picks up a stopped answer.** The note the agent gets about a turn stopped mid-answer changes:
  - on "continue", the agent goes on from where the response stopped and does not repeat the part the user already saw;
  - on "retry", it gives the full response, as today.
- **A failed turn saves the out-of-scope result.** Its state holds `out_of_scope` and `out_of_scope_reasoning`, as a successful turn's state does.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `chat-error-handling`:
  - The "Continue" button needs an agent tool call in the turn.
  - A rate-limited turn without one gets no button and no note.
  - The rate-limit message is followed by the Continue note only on a turn with the button.
  - A failed turn saves the out-of-scope result.
- `failed-turn-history`: the note after a turn stopped mid-answer tells the agent to continue from where the response stopped when the user asks to continue.
- `deep-research`: the scenario of a rate-limited turn without the Deep Research toggle now names the tool call the button needs. No behavior changes there.

## Impact

- Code:
  - `statgpt/app/application/channel_completion.py`: the decision on the button and the note, and the out-of-scope result on a failed turn.
  - `statgpt/app/chains/parameters.py`: the out-of-scope getters tolerate a check that didn't run.
  - `statgpt/app/default_prompts/assets/supreme_agent.yaml`: `interruptedTurnMessageToAgent`.
- Tests:
  - `tests/unit/app/application/test_channel_completion_state.py`
  - `tests/unit/app/utils/test_message_history.py`
  - Unit tests for the out-of-scope getters, in the existing file next to `statgpt/app/chains/parameters.py`, or a new one if there is none.
- Clients: no format change. DIAL Chat and StatGPT Portal Frontend already handle a failed message with and without the button.
- API: the error's status, code and headers don't change. Only the presence of the button and the note does.
- Branch: this goes into `feat/831-rate-limit-continue-button`, before the feature is released, so no user sees the earlier behavior.

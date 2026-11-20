# Tasks

## 1. Decide the button by the agent's tool calls

- [x] 1.1 In `ChannelCompletion._channel_completion` (`statgpt/app/application/channel_completion.py`), add a helper that says whether the turn has something to continue.
  - True when `ParamsConfig.HISTORY` is in `inputs` and `History.get_tool_messages()` holds an `AIMessage` with `tool_calls`.
  - Use it in `continuable`: streamed, a rate limit or a Deep Research deployment failure, and something to continue.
  - Update the comment on `resumable_failure` and on `CONTINUE_NOTE` to match.

- [x] 1.2 Extend `_patch_chain` / `_run_turn` in `tests/unit/app/application/test_channel_completion_state.py` so a failing turn can first add tool messages to the history and append content to the choice. Verify with tests in that file:
  - streamed rate limit after an agent tool call: button, note, and the full text in `error_display_message`;
  - streamed rate limit after a tool call with an error response: button;
  - streamed rate limit after content only: no button, no note;
  - streamed rate limit with no tool call and no content: no button, no note, and `error_display_message` holds the rate-limit text alone;
  - streamed rate limit with only a system note in the tool messages: no button;
  - streamed rate limit before the history was built: no button;
  - streamed out-of-scope refusal stopped by a rate limit: no button;
  - streamed Deep Research deployment failure, with and without a session: button (the failed call is in the history);
  - the existing tests that expect the button after a bare rate limit are updated to add a tool call first;
  - non-streaming, forbidden deployment and successful turns: unchanged.
  - `make test_unit` passes.

## 2. The out-of-scope result on a failed turn

- [x] 2.1 In `statgpt/app/chains/parameters.py`, make `ChainParameters.is_out_of_scope` and `get_out_of_scope_reasoning` read with `.get`, so a check that didn't run gives `None`.

- [x] 2.2 In `_channel_completion`, write `out_of_scope` and `out_of_scope_reasoning` into the state of every failed turn that sends its state, in one place after the `except` branches (see `design.md`).

- [x] 2.3 Verify with tests in `tests/unit/app/application/test_channel_completion_state.py`:
  - a rate limit after the check rejected the message: the state holds `out_of_scope` = `True` and the reasoning, and there is no button;
  - a rate limit before the check: both are `None`.

- [x] 2.4 Add unit tests for the two getters, in the existing test file for `statgpt/app/chains/parameters.py`, or a new one if there is none:
  - the key was never set: both return `None`;
  - the check ran: both return the saved values.
  - `make test_unit` passes.

## 3. The stopped-answer note

- [x] 3.1 In `statgpt/app/default_prompts/assets/supreme_agent.yaml`, rewrite the last paragraph of `interruptedTurnMessageToAgent` (see the `failed-turn-history` delta).
  - On "continue": go on from where the response stopped, without repeating the part above; an unfinished sentence, table, list or code block may be started again.
  - On "retry": give the full response.
  - Keep "even if it said to wait or to try again later".
  - Verify: tests in `tests/unit/app/utils/test_message_history.py` that format this note still pass, and one test asserts the note mentions both cases. `make test_unit` passes.

## 4. Checks

- [x] 4.1 Run `make format` and `make lint`; both pass.

- [x] 4.2 Check through local DIAL Core (`statgpt-sample`, streamed requests) and record the results in the PR description:
  - a rate limit after a data query: the button and the note are there;
  - a rate limit on the first model call: the plain rate-limit error, no button;
  - "Continue" after an answer stopped mid-way after a data query: the new answer goes on and does not repeat the shown part.
  - If a rate limit can't be forced locally, say so in the PR and rely on the unit tests.

## Workflow follow-up

- Archive the change after review, so the deltas are merged into `chat-error-handling`, `failed-turn-history` and `deep-research`.

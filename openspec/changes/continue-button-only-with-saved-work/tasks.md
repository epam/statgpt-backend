# Tasks

## 1. Track content sent to the user

- [ ] 1.1 In `statgpt/app/utils/dial_stages.py`, add a helper that wraps a choice's `append_content` and reports whether non-empty content was sent (see `design.md`, "Tracking content").
  - The wrapped choice behaves as before for every caller.
  - Verify: unit tests next to the existing `dial_stages` tests cover: no content, empty text only, and non-empty text. `make test_unit` passes.

## 2. Decide the button by what the turn holds

- [ ] 2.1 In `ChannelCompletion._channel_completion` (`statgpt/app/application/channel_completion.py`), install the content tracker right after the choice is created.

- [ ] 2.2 Add a helper in the same file that says whether the turn has something to continue.
  - True when `History.get_tool_messages()` holds an `AIMessage` with `tool_calls`, if `ParamsConfig.HISTORY` is in `inputs`.
  - True when the tracker says content was sent.
  - Use it in `continuable`: streamed, a rate limit or a Deep Research deployment failure, and something to continue.
  - Update the comment on `resumable_failure` and on `CONTINUE_NOTE` to match.

- [ ] 2.3 Extend `_patch_chain` / `_run_turn` in `tests/unit/app/application/test_channel_completion_state.py` so a failing turn can first add tool messages to the history and append content to the choice. Verify with tests in that file:
  - streamed rate limit after an agent tool call: button, note, and the full text in `error_display_message`;
  - streamed rate limit after a tool call with an error response: button;
  - streamed rate limit after content only: button;
  - streamed rate limit with no tool call and no content: no button, no note, and `error_display_message` holds the rate-limit text alone;
  - streamed rate limit with only a system note in the tool messages: no button;
  - streamed rate limit before the history was built: no button;
  - streamed Deep Research deployment failure, with and without a session: button (the failed call is in the history);
  - the existing tests that expect the button after a bare rate limit are updated to add a tool call first;
  - non-streaming, forbidden deployment and successful turns: unchanged.
  - `make test_unit` passes.

## 3. The stopped-answer note

- [ ] 3.1 In `statgpt/app/default_prompts/assets/supreme_agent.yaml`, rewrite the last paragraph of `interruptedTurnMessageToAgent` (see the `failed-turn-history` delta).
  - On "continue": go on from where the response stopped, without repeating the part above; an unfinished sentence, table, list or code block may be started again.
  - On "retry": give the full response.
  - Keep "even if it said to wait or to try again later".
  - Verify: tests in `tests/unit/app/utils/test_message_history.py` that format this note still pass, and one test asserts the note mentions both cases. `make test_unit` passes.

## 4. Checks

- [ ] 4.1 Run `make format` and `make lint`; both pass.

- [ ] 4.2 Check through local DIAL Core (`statgpt-sample`, streamed requests) and record the results in the PR description:
  - a rate limit after a data query: the button and the note are there;
  - a rate limit on the first model call: the plain rate-limit error, no button;
  - "Continue" after an answer stopped mid-way: the new answer goes on and does not repeat the shown part.
  - If a rate limit can't be forced locally, say so in the PR and rely on the unit tests.

- [ ] 4.3 Update the description of PR #832 and the text of issue #831: the button appears only when the turn has a tool call or content, and "Continue" picks up a stopped answer. Verify the texts name no client.

## Workflow follow-up

- Archive the change after review, so the deltas are merged into `chat-error-handling` and `failed-turn-history`.

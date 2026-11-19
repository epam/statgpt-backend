# Tasks

## 1. Rate-limit message

- [ ] 1.1 Change `RateLimitException._get_display_message` in `statgpt/app/utils/dial_exceptions.py` to the core message from the `chat-error-handling` delta.
  - The exceeded limits, then "The limit resets in <duration>.", or "Please wait for the limit to reset." without `Retry-After`.
  - No "try again" and no streaming flag.
  - Verify: tests in `tests/unit/app/utils/test_dial_exceptions.py` cover the daily-limit text with `Retry-After: 3700`, the text without `Retry-After`, and the text without known limits. `make test_unit` passes.

- [ ] 1.2 Change `_DEEP_RESEARCH_ERROR_TEXT` in `statgpt/app/chains/deep_research/errors.py` to "Deep Research could not complete this request.".
  - Verify: update the texts expected in `tests/unit/app/chains/test_deep_research_routing.py`, `tests/unit/app/application/test_channel_completion_state.py` and `tests/unit/app/utils/test_message_history.py`. `make test_unit` passes.

- [ ] 1.3 In `statgpt/app/default_prompts/assets/supreme_agent.yaml`, change "even if it said to try again later" to "even if it said to wait or to try again later" in `failedTurnWithErrorMessageToAgent` and `interruptedTurnMessageToAgent`.
  - Verify: update any test in `tests/unit/app/utils/test_message_history.py` that expects the old wording. `make test_unit` passes.

## 2. Form schema with the Continue button

- [ ] 2.1 In `statgpt/app/services/chat_facade.py`, add a message form-schema builder.
  - Inputs: an optional `deep_research` value and a flag for the "Continue" button.
  - The button: property `button`, `Button(const="continue", title="Continue", submit=True, populateText="Continue")`, no description.
  - Returns `None` when both are absent.
  - Keep the toggle-only output identical to today's `build_deep_research_form_schema`.
  - Verify: tests in `tests/unit/app/application/test_deep_research_toggle_form_schema.py` (or a sibling test file) cover:
    - toggle only, unchanged shape;
    - button only;
    - both properties in one schema;
    - `None` when both are absent.

## 3. Continuable turns in `ChannelCompletion`

- [ ] 3.1 In `ChannelCompletion._channel_completion` (`statgpt/app/application/channel_completion.py`), decide once whether the turn is continuable: `request.stream` is true, and the turn caught `openai.RateLimitError` or `DeepResearchFailedError`.
  - If it is, append the Continue note (one module constant, the exact text from the delta) to the DIAL error's `display_message` after a blank line, before `error_display_message` is saved.
  - Replace `_emit_deep_research_form_schema` with one emission that calls `set_form_schema` at most once. It includes the toggle when Deep Research is available to the caller, and the button when the turn is continuable.
  - Update `_patch_failing_chain` / `_run_failing_turn` in `tests/unit/app/application/test_channel_completion_state.py`: set `request.stream` explicitly and patch the renamed emitter.
  - Verify with tests in that file:
    - streamed rate limit and streamed Deep Research failure (with and without a session): the button is in the schema, the note ends `display_message`, and the state's `error_display_message` holds the full text;
    - a streamed rate limit where Deep Research is unavailable: a schema with the button only;
    - the same errors non-streaming: no button and no note;
    - a forbidden deployment and a successful turn: no button;
    - `test_failed_turn_keeps_the_error_shown_to_the_user` expects the new texts.

## 4. Save tool messages on a rate limit

- [ ] 4.1 In the `openai.RateLimitError` branch of `ChannelCompletion._channel_completion`, call `History.dump_state(state)` when the history was built, as the Deep Research branch does.
  - Verify with a test in `tests/unit/app/application/test_channel_completion_state.py`: run a rate-limited turn whose history holds a tool call and response, and assert the state's `tool_messages` contain them.
  - Update the rate-limit texts in the existing tests of `tests/unit/app/utils/test_message_history.py` to the new message.
  - Verify with a test in `tests/unit/app/utils/test_message_history.py`: cover the "Rate limit after a tool call" scenario of the `failed-turn-history` delta, i.e. tool messages, then the partial answer, then the interrupted-turn note.
- [ ] 4.2 Cover the "Rate limit after Deep Research replied" scenario.
  - Verify with a test that makes the mediation loop's second model call raise `openai.RateLimitError` after a resume call returned a clarification. Assert the state holds both the updated `deep_research_session` and the resume call with its response in `tool_messages`.

## 5. Data attachments on a failed turn

- [ ] 5.1 In `SupremeAgentExecutor.stream_response` (`statgpt/app/chains/supreme_agent.py`), add the data attachments of the collected artifacts on every way out of the agent loop: the final answer (as today), the iteration cap and an exception.
  - On an exception, add them and raise the exception again. Log and drop a failure of the attachments themselves.
  - Handle `Exception` only, so a cancelled turn gets none.
  - Verify with tests of the agent loop:
    - a Data Query call returns rows and the next model call raises `openai.RateLimitError`: the attachments are added and the same error is raised;
    - the same, with the displayer raising: the rate-limit error is still the one raised;
    - the iteration cap is reached after a query: the attachments are added;
    - a successful turn adds them once.

## 6. Integration check

- [ ] 6.1 Run `make format` and verify that `make lint` and `make test_unit` pass.
- [ ] 6.2 Check the change end to end through local DIAL Core (`http://localhost:8080`, deployment `statgpt-sample`).
  - Use a streamed request whose agent model is rate-limited, e.g. a temporary low token limit in the local DIAL Core config, restored afterwards.
  - Verify: the SSE stream has the partial content, a `form_schema` with `button` (and `deep_research` if available), the state with `tool_messages`, the data attachments of the finished queries, and then the error with the core message and the Continue note.
  - Repeat with `stream: false` and verify the error has the core message only.
  - Then ask the user to check in DIAL Chat that "Continue" shows next to the error and finishes the answer without rerunning the tools, with the tables and graphs on the failed message above it.

## Workflow follow-up

- Archive the change after review, so the delta specs merge into `openspec/specs/chat-error-handling`, `openspec/specs/data-query-agent-response`, `openspec/specs/deep-research` and `openspec/specs/failed-turn-history`.

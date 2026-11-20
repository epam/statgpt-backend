# Design

## Context

See `proposal.md` for the motivation.

- **The decision point.** `ChannelCompletion._channel_completion` sets `resumable_failure` in the `openai.RateLimitError` and `DeepResearchFailedError` branches. `continuable = resumable_failure and request.stream` then adds the Continue note and the button.
- **Where a rate limit can hit before the agent calls a tool:**
  - the out-of-scope check and its streamed refusal;
  - the configured ("fake") tool calls, which run before every turn;
  - the first model call of the agent loop or of the Deep Research mediation loop;
  - an answer the agent writes without calling a tool.
- **What the history holds.** `History.get_tool_messages()` returns this turn's messages only: the agent's tool calls (`AIMessage` with `tool_calls`, with any preamble kept in it), their responses, and system notes.
  - System notes can be there without any tool call, for example the note that Deep Research is enabled.
  - Fake tool calls are not there. They go into a separate history that is prepended as DIAL messages.
  - A tool call is added only after the model's response is complete, and a tool's own failure becomes an error tool response. So every call has a response, except a failed Deep Research call, which `_record_failed_call` answers with an error.
- **A Deep Research deployment failure always has a tool call.** `DeepResearchFailedError` is raised only inside a Deep Research tool call, after the call was added to the history.
- **Content.** Text reaches the user through `choice.append_content` from several writers: the agent's answer and preamble, the Deep Research report, the query-check deactivation message, the out-of-scope refusal. It is not kept in `History`, which stores only what the agent sees again in this turn.
- **The out-of-scope result.** `MainChain._update_state` writes `out_of_scope` and `out_of_scope_reasoning`; a failed turn never reaches it. `_count_out_of_scope_msgs` counts assistant messages whose state has `out_of_scope` = `True`. `ChainParameters.is_out_of_scope` reads the key directly; it is missing when the turn fails before the check.
- **The next turn.** `History._failed_turn_note` adds `interruptedTurnMessageToAgent` after a failed assistant message that has content. Today it tells the agent to "give the full response now".

## Goals / Non-Goals

**Goals:**
- The button and the note appear only when the note's claim, "what was already done is kept", is true for this turn.
- One simple rule: an agent tool call, with no special case per tool.
- "Continue" after a stopped answer does not repeat the part already shown.
- An out-of-scope refusal stopped by an error stays rejected and counted.

**Non-Goals:**
- Judging how useful the saved work is: an error tool response and a failed Deep Research call count like any other call.
- Reading Deep Research's state to tell whether its snapshot can be resumed.
- Telling the user, on a turn without the button, that an earlier failed turn's work is still saved.
- A different note for a "Continue" turn whose Deep Research call fails again without a new state (see Risks).
- Changing the rate-limit text or the Deep Research fallback text.
- Keeping a partial answer written without tool calls: such a turn gets no button.

## Decisions

### The rule: an agent tool call
`continuable` becomes: the request is streamed, the turn ended with a rate limit or a Deep Research deployment failure, and the turn has something to continue: any `AIMessage` with `tool_calls` in `History.get_tool_messages()`, when the history was built.

The check is one small helper in `channel_completion.py`. Both failure branches go through it; for a Deep Research failure it is always true, so that branch needs no code of its own.
- Alternative: count only successful tool responses. Rejected by the user: a call without a useful response is still context for the agent, a failed Deep Research call can leave a state worth resuming, and "any call" needs no special cases.
- Alternative: no button for a Deep Research failure that sent no state. Rejected for the same reason (see Risks).
- Alternative: check that the list of tool messages is not empty. Rejected: system notes alone would count.
- Alternative: content sent to the user also counts. Rejected: content is not kept in `History`. Detecting it would mean patching the SDK choice, wrapping it (a wrapper fails `send_annotations`'s `isinstance(choice, Choice)` check), or having every writer record it. A turn with content only loses just the text, which the agent writes again.

### Out-of-scope turns
No check of their own: the main chain is skipped, so they have no agent tool call and get no button.

### The out-of-scope result in a failed turn's state
`_channel_completion` writes `out_of_scope` and `out_of_scope_reasoning` from `inputs` in one place after the `except` branches, for every turn that sends its state. `ChainParameters.is_out_of_scope` and `get_out_of_scope_reasoning` read with `.get`, so a check that didn't run gives `None`.
- Alternative: only on rate limits. Rejected: every stopped refusal should count.

### The stopped-answer note
`interruptedTurnMessageToAgent` in `supreme_agent.yaml` gets two instructions instead of one:
- the user asks to continue: go on from where the response stopped, without repeating the part above; an unfinished sentence, table, list or code block may be started again;
- the user asks to retry: give the full response.

The rest of the note stays: what the user saw, the quoted error, and that the error may be resolved.
- Alternative: keep "give the full response". Rejected: on a turn with tool calls and a partial answer, "Continue" would repeat the shown part.

### Texts on a turn without the button
The rate-limit message stays "… The limit resets in <duration>." with no "try again". The same text is used with and without the note, and the "Try again" button is next to it.

## Risks / Trade-offs

- [A "Continue" turn calls Deep Research, and it fails again without a new state] → The turn holds a tool call, so it gets the button and the note, though "Try again" would also resume from the earlier snapshot. The advice is cautious, never harmful. If Deep Research's own text says "Try again" is fine, the two texts disagree. Accepted by the user.
- [Deep Research sent a state it marks as not resumable] → The button and "what was already done is kept" are shown, and the research starts again on "Continue". The claim holds for the history, not for the research. Accepted: the state is opaque to StatGPT.
- [All tool responses are errors, e.g. the tools hit the same rate limit] → The button is shown. "Continue" gives the agent the failed calls, and it calls the tools again. Accepted with the "any call" rule.
- [An answer written without tool calls stopped near its end] → No button. "Try again" writes the whole answer again, costing its tokens. A user who types "continue" still gets the rest, thanks to the note. Accepted.
- [The continued answer starts mid-sentence or splits a table] → The note lets the agent start the unfinished block again.
- [No button on a turn that follows an earlier failed turn with saved work] → Nothing tells the user that the work is still saved. "Try again" resends the last message, and the earlier message keeps its state, so nothing is lost.

## Migration Plan

No data migration. The change lands on the feature branch before release. Roll back by reverting.

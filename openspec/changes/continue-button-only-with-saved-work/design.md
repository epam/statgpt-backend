# Design

## Context

See `proposal.md` for the motivation.

- **The decision point.** `ChannelCompletion._channel_completion` sets `resumable_failure` in the `openai.RateLimitError` and `DeepResearchFailedError` branches. `continuable = resumable_failure and request.stream` then adds the Continue note and the button.
- **Where a rate limit can hit before the agent does anything:**
  - the out-of-scope check;
  - the configured ("fake") tool calls, which run before every turn;
  - the first model call of the agent loop or of the Deep Research mediation loop.
- **What the history holds.** `History.get_tool_messages()` returns this turn's messages only: the agent's tool calls (`AIMessage` with `tool_calls`), their responses, and system notes.
  - System notes can be there without any tool call, for example the note that Deep Research is enabled.
  - Fake tool calls are not there. They go into a separate history that is prepended as DIAL messages.
  - A tool call is added only after the model's response is complete, and a tool's own failure becomes an error tool response. So every call has a response, except a failed Deep Research call, which `_record_failed_call` answers with an error.
- **A Deep Research deployment failure always has a tool call.** `DeepResearchFailedError` is raised only inside a Deep Research tool call, after the call was added to the history.
- **Content.** Text reaches the user through `choice.append_content`: the agent's answer, a preamble before a tool call, the Deep Research report, the query-check deactivation message. The SDK's `Choice` has no public way to ask whether content was sent.
- **The next turn.** `History._failed_turn_note` adds `interruptedTurnMessageToAgent` after a failed assistant message that has content. Today it tells the agent to "give the full response now".

## Goals / Non-Goals

**Goals:**
- The button and the note appear only when the note's claim, "what was already done is kept", is true for this turn.
- One simple rule, with no special case per tool.
- "Continue" after a stopped answer does not repeat the part already shown.

**Non-Goals:**
- Judging how useful the saved work is: an error tool response and a failed Deep Research call count like any other call.
- Reading Deep Research's state to tell whether its snapshot can be resumed.
- Telling the user, on a turn without the button, that an earlier failed turn's work is still saved.
- A different note for a "Continue" turn whose Deep Research call fails again without a new state (see Risks).
- Changing the rate-limit text or the Deep Research fallback text.

## Decisions

### The rule: an agent tool call or content
`continuable` becomes: the request is streamed, the turn ended with a rate limit or a Deep Research deployment failure, and the turn has something to continue:
- any `AIMessage` with `tool_calls` in `History.get_tool_messages()`, when the history was built;
- or any content appended to the choice in this turn.

The check is one small helper in `channel_completion.py`. Both failure branches go through it; for a Deep Research failure it is always true, so that branch needs no code of its own.
- Alternative: count only successful tool responses. Rejected by the user: a call without a useful response is still context for the agent, a failed Deep Research call can leave a state worth resuming, and "any call" needs no special cases.
- Alternative: no button for a Deep Research failure that sent no state. Rejected for the same reason (see Risks).
- Alternative: check that the list of tool messages is not empty. Rejected: system notes alone would count.

### Tracking content
Record whether `choice.append_content` was called with non-empty text in this turn. The helper lives next to the other choice helpers in `statgpt/app/utils/dial_stages.py`: it wraps the choice's `append_content` and returns a way to ask whether content was sent. `_channel_completion` installs it right after the choice is created.
- All content counts, whoever wrote it. This matches `failed-turn-history`, which treats any content of a failed message as a stopped answer.
- Alternative: read the SDK's private `_last_finish_reason`. Rejected: it is private and can change with the SDK.
- Alternative: a flag set only where the Supreme Agent streams its answer. Rejected: other writers would be missed, and the next turn's note would disagree with the button.

### The stopped-answer note
`interruptedTurnMessageToAgent` in `supreme_agent.yaml` gets two instructions instead of one:
- the user asks to continue: go on from where the response stopped, without repeating the part above; an unfinished sentence, table, list or code block may be started again;
- the user asks to retry: give the full response.

The rest of the note stays: what the user saw, the quoted error, and that the error may be resolved.
- Alternative: keep "give the full response". Rejected: a button on a turn with content only would then save nothing.

### Texts on a turn without the button
The rate-limit message stays "… The limit resets in <duration>." with no "try again". The same text is used with and without the note, and the "Try again" button is next to it.

## Risks / Trade-offs

- [A "Continue" turn calls Deep Research, and it fails again without a new state] → The turn holds a tool call, so it gets the button and the note, though "Try again" would also resume from the earlier snapshot. The advice is cautious, never harmful. If Deep Research's own text says "Try again" is fine, the two texts disagree. Accepted by the user.
- [Deep Research sent a state it marks as not resumable] → The button and "what was already done is kept" are shown, and the research starts again on "Continue". The claim holds for the history, not for the research. Accepted: the state is opaque to StatGPT.
- [All tool responses are errors, e.g. the tools hit the same rate limit] → The button is shown. "Continue" gives the agent the failed calls, and it calls the tools again. Accepted with the "any call" rule.
- [Only the query-check deactivation message was sent before the rate limit] → It counts as content, so the button is shown and the agent treats the message as a stopped answer. Rare, and "Continue" still works.
- [The continued answer starts mid-sentence or splits a table] → The note lets the agent start the unfinished block again.
- [No button on a turn that follows an earlier failed turn with saved work] → Nothing tells the user that the work is still saved. "Try again" resends the last message, and the earlier message keeps its state, so nothing is lost.

## Migration Plan

No data migration. The change lands on the feature branch before release. Roll back by reverting.

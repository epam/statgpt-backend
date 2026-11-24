# Design

## Context

See proposal.md (Why) for the failure. The relevant parts as they work today:

- **Mediation loop**: `SupremeAgentChain._run_deep_research_turn` (`statgpt/app/chains/supreme_agent.py`). Each pass runs the agent inside an empty `[DEBUG] Deep Research mediation run N` stage. Then it either executes the tool calls or returns the agent's text as the answer. On a resume turn, the agent only sees the resume tool. The instructions are the `deepResearchSection` prompt (`supreme_agent.yaml`, not channel-overridable), the resume tool description (`DeepResearchDetails.resume_tool_description`, channel-overridable, default in `tool_details.py`) and the `message` argument description (`ResumeDeepResearchArgs`).
- **Prompt gaps**: The prompt only names two things to send through the resume tool: answers to clarifying questions and approval of the plan. It tells the agent to redirect "unrelated" requests and to present plans "verbatim". It never says that amendments go to Deep Research or that the agent must not write a plan itself. In the export, the agent did three wrong things: it called "please also check the annual risk reports" unrelated, it re-presented the earlier plan from history, and it wrote an amended plan itself. All three fit these gaps.
- **Resume stage names**: `ResumeDeepResearchTool._render_stage_name` fills `{original_question}` from `DeepResearchSession.original_question`, which is `turns[0].user_message`. That is the message the agent sent to the start tool, not Deep Research's query. `ToolCaller.call_tool` renders both the call name and the result name **before** the tool runs. The result stage is a `DelayedStage` that only opens on first content, but its name is fixed when it is created.
- **What each side sees**: the agent's history is built from every message of the request (`History.from_dial_with_interceptors`), so it includes the turns from before Deep Research mode was turned on. Deep Research gets only the session's replayed turns plus the new message (`DeepResearchRunner._build_request_messages`). A reference such as "the publication I asked about" is therefore resolvable by the agent but not by Deep Research. The prompt only says to make the resume message "clear and self-contained", and does not say that Deep Research cannot see the chat or that references must be written out.
- **Deep Research state**: each stored turn holds Deep Research's state. Its `preparation.current_query` is the query that `update_query` last set (see the export: `"current_query": "Compare the provider's US policy-rate forecast for 2027 versus a suitable consensus forecast source."`). We already read `preparation.research_started` from the same object.

## Goals / Non-Goals

**Goals:**
- The mediating agent forwards query and plan changes to Deep Research. This is a prompt-only fix.
- Every message the agent sends to Deep Research, the research question included, carries the context it needs from the conversation.
- A developer can see from the debug stages exactly what the agent sent to Deep Research in each mediation run.
- Resume stage names show Deep Research's current query.

**Non-Goals:**
- No code-level enforcement, for example forcing the resume call or detecting plan-like text in the agent's answer. The requester chose prompt-only.
- No change to how Deep Research handles amendments. That belongs to the Deep Research repository.
- No change to the start tool's stage names. They already use the `query` argument of that call.
- No change to what Deep Research receives, e.g. replaying the chat from before the session to it. Deep Research would have to tell earlier chat apart from session turns, and the agent already holds that chat.

## Decisions

### D1. Prompt: route by topic, forward when unsure, never author a plan
Rewrite the `deepResearchSection` bullets so the agent decides with one rule:
- **Forward**: anything about the pending question, the query or the plan goes through the resume tool. That covers answers, approval, amendments (add, remove or change steps, sources, publication types, scope, period, the question itself) and questions or objections about the plan, including "you did not add X".
- **Redirect**: only requests clearly unrelated to the research. When unsure, forward.
- **Plans**: a plan shown to the user must be the one Deep Research returned in the current turn's tool response, copied verbatim. The agent never writes, edits, merges or re-presents a plan from an earlier turn. If the user wants a change, the agent forwards it and shows whatever Deep Research returns.
- Add the counter-example from the export to the prompt: "please also check the annual risk reports" while a plan is pending is an amendment, not an unrelated request.

Update the default `resume_tool_description` and the `message` field description the same way: the message carries answers, approval, **or requested changes to the query or plan**. Alternative: change only the system prompt. Rejected, because the tool and argument descriptions currently list only "answers or approval". That steers the model away from the tool for amendments.

Keep the existing "do not narrate / either call the tool or write" rule.

### D1a. Prompt: make every message to Deep Research self-contained
Replace the "compose the resume message so it is clear and self-contained" bullet with a rule that states why: Deep Research does not see the chat, only the research question and the agent's messages. The agent resolves the user's references to the conversation, including turns from before Deep Research mode was turned on, into concrete details: publication titles (and type or date where known), datasets, indicators, countries, periods, figures and earlier answers. The example in the prompt is "add the publication I asked about to the list", which must become a message that names the publication. When a reference could mean more than one thing, the agent asks the user instead of guessing. The rule covers the start question too, so the start tool's `query` description says the same, as do the `message` description and the default `resume_tool_description`.

The "never invent" rule stays and is widened to details: the agent only uses what the user or the conversation provided.

Alternative: replay the chat from before the session to Deep Research. Rejected, see Non-Goals.

### D2. Mediation debug stage shows the decision
After `agent.run` returns, write to the open mediation stage:
- for each tool call: the tool name and its arguments as a fenced JSON block (`json.dumps(args, ensure_ascii=False, indent=2)`), like the existing `[TOOL] … **Input**` stages;
- with no tool calls: a one-line note that the agent answered the user directly.

Write it inside the existing `optional_timed_stage` block, so it is only shown with debug stages on. Alternative: put the args in the `Researching: …` stage. Rejected, because that stage is not debug-only and users would see internal composed messages.

### D3. `{query}` = latest `preparation.current_query`, with fallback
Add `DeepResearchSession.latest_query`: go through `turns` from newest to oldest and return the first non-empty string at `deep_research_state['preparation']['current_query']`, else `original_question`. Read the field defensively, like `_research_started`: a missing or non-dict value means "not present". Expose it as the `{query}` placeholder in `ResumeDeepResearchTool._render_stage_name`, next to `{original_question}`. The resume tool has no `query` argument, so the name does not clash with tool args. If neither value is available and the template references `{query}` or `{original_question}`, use `missing_question_name`.

Change the `ResumeStagesConfig` defaults to `Researching: {query}` / `Research result: {query}` and update their descriptions. Alternative: redefine `{original_question}` as the latest query. Rejected, because the name would then be misleading and channels that want the start question would lose it.

### D4. Name the result stage after the call
In `ToolCaller.call_tool`, render the result stage's name when the result stage opens, not when the call starts. `delayed_timed_stage` already creates the real stage lazily, through `DelayedStage(lambda: choice.create_stage(...))`. Let it (and `optional_delayed_timed_stage`) also accept a zero-argument callable as the name, called inside that lambda. `call_tool` then passes `lambda: prefix + tool.render_result_stage_name(args, inputs)`, using the args from before `inputs` is injected. `DeepResearchRunner.run` saves the updated session to state **before** it calls `target.append_content(content)`, so the result stage opens after the query has been rewritten. That means reordering `_append_turn` / `_save_session` before `append_content` in the clarification branch.

Alternative: rename the stage after it closes. Rejected, because DIAL stage names are appended, not replaced, so a rename would concatenate. Second alternative: leave the result stage named before the call. Rejected, because the user wants the latest query, and the turn where `update_query` rewrites the query is exactly when it matters.

The change to `call_tool` applies to every tool, but other tools' names depend only on their args, which do not change during the call, so their rendered names stay the same.

## Risks / Trade-offs

- [The agent may resolve a reference wrongly, e.g. pick the wrong one of two publications] → Mitigation: the prompt tells it to ask the user when a reference is ambiguous, and the mediation debug stage (D2) shows the exact message sent.
- [The fix is prompt-only, so the model can still skip the resume call] → Mitigation: D2 makes a skip visible in debug stages at once. Add routing tests that cover the forward path for amendments. A code guard is a possible follow-up if the problem comes back.
- [Forwarding when unsure may send borderline off-topic messages to Deep Research, which could rewrite its query] → Mitigation: the redirect rule stays for clearly unrelated requests. Deep Research's preparation agent already checks query changes.
- [Channels that override `resume_tool_description` keep the old "answers or approval" wording] → Mitigation: mention it in the PR / release notes. The system prompt section, which is not overridable, still carries the rule.
- [Channels that override `resumeStagesConfig` with `{original_question}` keep stale names] → This is intended and documented. They switch to `{query}` to opt in.
- [`preparation.current_query` is a Deep Research internal field] → Mitigation: falls back to the original question. It is listed next to `research_started` as part of the contract.
- [Debug stages show the agent's composed messages, which may quote user text] → Debug stages are already off for regular users, and the `[TOOL]` stages already show the same kind of data.

## Migration Plan

No data migration is needed. Old sessions without `current_query` fall back to the original question. Rollback means reverting the commit. Stored state is not changed in shape.

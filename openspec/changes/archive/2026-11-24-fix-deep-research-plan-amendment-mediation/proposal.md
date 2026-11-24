# Proposal

## Why

While a Deep Research plan waits for approval, the mediating Supreme Agent does not forward requests to change the plan. In an exported chat (a request to compare a data provider's 2027 US policy-rate forecast with the consensus):

- "please also check the annual risk reports" was treated as unrelated to the session. The agent asked the user to "approve or amend the plan", which is what the user had just done.
- "i want to add a plan step to check the annual risk reports" got the unchanged plan from the previous turn back, without calling Deep Research.
- "you did not add the risk reports" got a plan that the agent rewrote itself. It looked right, but Deep Research never received it, so its stored query and plan stayed unchanged. Deep Research only uses a query and plan made by its own preparation tools.

Only the Deep Research Preparation flow (the `update_query` / `update_plan` tools behind the resume tool) can change the query or plan, so the user cannot amend a plan at all.

Forwarding the user's words as they are is not enough either. Deep Research does not see the chat: it sees only the research question and the messages the agent sends it. A user may ask about a publication before turning Deep Research on, get an answer from the agent, and later say "add the publication I asked about to the list". Deep Research cannot tell which publication that is, unless the agent writes out the publication it means. In the debug view, this was hard to diagnose: the `[DEBUG] Deep Research mediation run N` stages are empty, so they do not show whether the agent called the resume tool or what it sent. The `Researching: …` stage name also shows the question the session started with. After Deep Research rewrites the query, that name is out of date.

## What Changes

- The mediation prompt (`deepResearchSection`), the resume tool's description and the `message` / `query` argument descriptions will say:
  - any reply about the pending question, query or plan goes to Deep Research through the resume tool: answers, approval, and also amendments (add, remove or change steps, sources, scope or the question) and questions about the plan;
  - every message to Deep Research, the research question included, is self-contained: the agent resolves references to the conversation, including turns from before Deep Research mode was turned on, into concrete details (e.g. the title of a publication discussed earlier), and asks the user when a reference is ambiguous;
  - the agent never writes, edits or re-presents a plan itself; a plan shown to the user is the one Deep Research returned in this turn;
  - only clearly unrelated requests are redirected, and when unsure, the agent forwards the message.

  This is a prompt-only fix. No code check enforces it (decided with the requester).
- Each `[DEBUG] Deep Research mediation run N` stage will show what the agent decided: for each tool call, the tool name and its arguments (for example the `message` sent to Deep Research), or a note that the agent answered the user directly.
- The resume tool's stage names will show the session's **latest** query: the `preparation.current_query` of the last stored Deep Research state. A new `{query}` placeholder carries it. It falls back to the original question when Deep Research has not stored a query. The defaults change to `Researching: {query}` / `Research result: {query}`. `{original_question}` stays supported with its current meaning. The result stage's name is rendered once the call returns, so it shows the query as it stands after this turn.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `deep-research`: mediation must forward plan and query amendments to Deep Research and must not write plans itself ("Supreme Agent mediates clarifications and the plan"), and must make every message to Deep Research self-contained, using the conversation context (new requirement). The mediation debug stages show the agent's tool calls (new requirement). Resume stage names use the latest query ("Resume stage names", "Missing original question").

## Impact

- Prompts: `statgpt/app/default_prompts/assets/supreme_agent.yaml` (`deepResearchSection`). `DeepResearchArgs.query` and `ResumeDeepResearchArgs.message` descriptions in `statgpt/app/chains/deep_research/deep_research_tool.py`. Default `resume_tool_description`.
- Code: `SupremeAgentChain._run_deep_research_turn` in `statgpt/app/chains/supreme_agent.py` (debug stage body). `ResumeDeepResearchTool._render_stage_name`, `DeepResearchSession` (`statgpt/app/schemas/deep_research.py`), `ToolCaller.call_tool` (result stage named after the call).
- Config: `ResumeStagesConfig` defaults and descriptions in `statgpt/common/schemas/tool_details.py`. Channels that override `resumeStagesConfig` with `{original_question}` keep their current names until they switch to `{query}`.
- Contract with Deep Research: we now also read `preparation.current_query` from its state, as we already read `preparation.research_started`. If the field is missing, names fall back to the original question.
- Tests: `tests/unit/app/chains/test_deep_research_stage_names.py`, `tests/unit/app/chains/test_deep_research_routing.py`, `tests/unit/app/chains/test_deep_research_session.py`, `tests/unit/app/chains/test_supreme_agent_prompt.py`, `tests/unit/app/utils/test_dial_stages.py`.
- No API, schema-migration or frontend changes.

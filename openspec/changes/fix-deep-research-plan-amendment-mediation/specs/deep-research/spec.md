# Spec Delta

## MODIFIED Requirements

### Requirement: Supreme Agent mediates clarifications and the plan
A clarification or plan SHALL be shown in the tool-result stage and handed back to the agent, which SHALL answer the clarifying questions it can from the conversation context through the resume tool, surface to the user only the questions it cannot answer, and present a plan verbatim, approving it only once the user does. When the user asks for something unrelated to the research session during a session, the agent SHALL ask them to answer the pending question or to turn the mode off. A request to change the query or plan SHALL NOT count as unrelated.

#### Scenario: Questions partly answered from context
- **WHEN** Deep Research asks for the countries and the period, and the user already named the countries earlier in the chat
- **THEN** the agent sends the countries through the resume tool and asks the user only for the period

#### Scenario: Plan for approval
- **WHEN** Deep Research presents a research plan
- **THEN** the agent shows it to the user verbatim and the turn ends without approving it

#### Scenario: Unrelated request mid-session
- **WHEN** a plan awaits approval and the user asks "What datasets do you have?"
- **THEN** the agent does not call the resume tool and asks the user to answer the pending plan or to turn Deep Research mode off

### Requirement: Resume stage names
The resume tool's stage names SHALL come from `resumeStagesConfig` (`toolCallName`, default "Researching: {query}"; `toolResultName`, default "Research result: {query}"), never from the start tool's stage names, because the resume tool takes a `message` argument instead of `query`. They SHALL support the `{query}` placeholder, the session's latest research query, as well as `{original_question}`, the question the session was started with, and `{message}`. The latest query SHALL be the `preparation.current_query` of the latest Deep Research state stored in the session, falling back to the original question when no stored state carries one. The tool-call stage SHALL be named from the session as it was before the call, and the tool-result stage from the session as it is after the call, so the result stage reflects a query Deep Research rewrote in that call.

#### Scenario: Resume call
- **WHEN** a session started with "Drivers of inflation in Europe" is resumed and Deep Research has stored no query
- **THEN** the stage is named "Researching: Drivers of inflation in Europe"

#### Scenario: Query rewritten by Deep Research
- **WHEN** a session started with "Compare the provider's US policy rates forecast for 2027 versus the consensus." and Deep Research's latest stored state has `preparation.current_query` "Compare the provider's US policy-rate forecast for 2027 versus a suitable consensus forecast source."
- **THEN** the resume call stage is named "Researching: Compare the provider's US policy-rate forecast for 2027 versus a suitable consensus forecast source."

#### Scenario: Query rewritten during the call
- **WHEN** the resume call makes Deep Research rewrite its query from A to B
- **THEN** the tool-call stage is named after A and the tool-result stage after B

#### Scenario: Original question placeholder
- **WHEN** a channel configures `toolCallName` "Researching: {original_question}" and Deep Research has rewritten the query
- **THEN** the stage still shows the question the session was started with

### Requirement: Missing original question
When neither the latest query nor the original question can be read from the session, a resume stage name that references `{query}` or `{original_question}` SHALL be replaced entirely by `missingQuestionName` (default "Continue researching") rather than shown half-filled.

#### Scenario: No session in the state
- **WHEN** the resume tool is called and no session turn is stored
- **THEN** its stage is named "Continue researching"

## ADDED Requirements

### Requirement: Query and plan changes go through Deep Research
While a session is in progress, the agent SHALL forward through the resume tool every message about the pending question, query or plan: answers, approval, amendments and questions about the plan. When unsure, the agent SHALL forward it. Only Deep Research can update both the query and the plan, so the agent SHALL NOT write, edit or re-present either itself: a plan shown to the user SHALL be one Deep Research returned in the same turn.

#### Scenario: Plan amendment phrased as an addition
- **WHEN** a plan awaits approval and the user writes "please also check the annual risk reports"
- **THEN** the agent calls the resume tool with that request and presents the plan Deep Research returns

#### Scenario: Explicit request to add a plan step
- **WHEN** a plan awaits approval and the user writes "i want to add a plan step to check the annual risk reports"
- **THEN** the agent calls the resume tool rather than repeating the previous plan

#### Scenario: User says the change was not applied
- **WHEN** the user writes "you did not add the risk reports" after a plan without the annual risk reports
- **THEN** the agent forwards the request through the resume tool and does not write an amended plan itself

### Requirement: Messages to Deep Research use the conversation context
Deep Research sees only the research question and the messages the agent sends it, not the chat. The agent SHALL therefore make the research question and every resume message self-contained: it SHALL resolve the user's references to the conversation, including turns from before Deep Research mode was turned on, into concrete details. When a reference could mean more than one thing, the agent SHALL ask the user instead of guessing.

#### Scenario: Plan change refers to an earlier answer
- **WHEN** the user asked about a publication before turning Deep Research on, the agent named it "Global Outlook 2026", and, with a plan awaiting approval, the user writes "add the publication I asked about to the list"
- **THEN** the agent calls the resume tool with a message that names "Global Outlook 2026", not "the publication I asked about"

#### Scenario: Research question refers to an earlier answer
- **WHEN** the agent described an indicator in a normal turn, and the user turns Deep Research on and writes "research what drives that indicator"
- **THEN** the research question the agent sends to Deep Research names the indicator

#### Scenario: Ambiguous reference
- **WHEN** two publications were discussed earlier and the user writes "add the publication I asked about"
- **THEN** the agent asks the user which publication they mean and does not call the resume tool

### Requirement: Mediation debug stages show the agent's decision
With debug stages shown, each `[DEBUG] Deep Research mediation run N` stage SHALL show what the agent decided in that run: for each tool call, the tool's name and its arguments (e.g. the `message` sent to Deep Research), or, when the agent made no tool call, that it answered the user directly. The stage SHALL NOT be shown without debug stages.

#### Scenario: Resume call
- **WHEN** debug stages are shown and the agent calls the resume tool with message "Add a step to search the annual risk reports"
- **THEN** that run's stage shows the resume tool's name and the message argument

#### Scenario: Direct answer
- **WHEN** debug stages are shown and the agent writes to the user without a tool call
- **THEN** that run's stage says the agent answered the user directly

#### Scenario: Debug stages off
- **WHEN** debug stages are not shown
- **THEN** no mediation stage is shown

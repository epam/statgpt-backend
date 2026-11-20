# Spec Delta

## MODIFIED Requirements

### Requirement: A failed turn's note invites a retry
The notes `failedTurnMessageToAgent`, `failedTurnWithErrorMessageToAgent` and `interruptedTurnMessageToAgent` SHALL say that the error affected only that attempt and may have been resolved since, and SHALL tell the agent that if the user asks to continue or to retry, it answers the message now instead of repeating the error. For a turn stopped mid-answer, the note SHALL tell the agent to go on from where the response stopped when the user asks to continue, without repeating the part the user already saw, and to give the full response when the user asks to retry. The agent MAY start again an unfinished sentence, table, list or code block. The notes that quote an error SHALL say this holds even if the error said to wait or to try again later.

#### Scenario: Continue after an error that said to try again later
- **WHEN** a Deep Research turn failed, the next turn failed with `You've exceeded your daily token limit for the underlying deployment. The limit resets in 1 day.`, and the user then sends `continue` after the limit was lifted
- **THEN** the note after the failed turn tells the agent to answer the user's message now instead of repeating the error, so the agent resumes Deep Research rather than replying that it cannot continue for a day

#### Scenario: Two failed turns in a row
- **WHEN** two turns in a row failed before any answer, and the user then sends `continue`
- **THEN** the agent answers the user's earlier message rather than replying with an empty answer

#### Scenario: Continue after an answer stopped near its end
- **WHEN** a turn streamed three paragraphs of its answer, failed with a rate limit in the last sentence, and the user then sends `Continue`
- **THEN** the note tells the agent to go on from where the response stopped, so the agent writes the rest of the answer and does not repeat the three paragraphs

#### Scenario: Retry after an answer stopped mid-way
- **WHEN** a turn streamed part of its answer, failed with a rate limit, and the user then writes `retry`
- **THEN** the note tells the agent to give the full response

"""LLM helpers that let the Supreme Agent mediate the Deep Research clarification flow:
summarize the conversation for Deep Research context, and triage Deep Research's clarifying
questions into ones the agent can answer from context vs. ones the user must answer."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, SecretStr

from statgpt.common.config import multiline_logger as logger
from statgpt.common.schemas import LLMModelConfig
from statgpt.common.utils.models import get_chat_model

_SUMMARY_SYSTEM_PROMPT = """\
You condense a chat conversation into a briefing for a Deep Research assistant.

Write a concise summary (a few short paragraphs or bullet points) that captures everything a \
researcher would need to understand and scope the user's request: the research goal, the specific \
entities/topics involved, any constraints (time range, geography, sources, format), and any \
preferences or details the user has already stated. Do not invent information that is not present \
in the conversation. Do not add commentary or ask questions — output only the summary.\
"""

_TRIAGE_SYSTEM_PROMPT = """\
A Deep Research assistant asked the user the clarifying questions below before starting its \
research. You decide, for each question, whether it can be answered CONFIDENTLY and UNAMBIGUOUSLY \
from the conversation context alone, so the user does not have to be asked again for information \
they already provided.

Rules:
- Only set can_answer=true when the context clearly and specifically contains the answer. When in \
doubt, set can_answer=false so the user is asked.
- Never guess, assume defaults, or infer preferences that the user did not express.
- When can_answer=true, put the answer in `answer`, phrased as a direct reply to that question.
- Return exactly one item per question, using the question's zero-based index.

Conversation context:
{context}

Clarifying questions (index: question):
{questions}\
"""


class _QuestionTriage(BaseModel):
    index: int = Field(description="Zero-based index of the question this decision refers to.")
    can_answer: bool = Field(
        description="True only if the question can be confidently answered from the context alone."
    )
    answer: str | None = Field(
        default=None, description="The answer when can_answer is true, else null."
    )


class _TriageResponse(BaseModel):
    items: list[_QuestionTriage] = Field(default_factory=list)


def build_transcript(messages: Sequence[BaseMessage]) -> str:
    """Render the user/assistant turns of a conversation as a plain-text transcript.

    Tool calls, tool results, and system messages are skipped: they are internal plumbing, not
    part of what the user and assistant said to each other."""
    lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage) and not msg.tool_calls:
            role = "Assistant"
        else:
            continue
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        if text.strip():
            lines.append(f"{role}: {text.strip()}")
    return "\n\n".join(lines)


async def summarize_conversation(
    *, api_key: str | SecretStr, model_config: LLMModelConfig, transcript: str
) -> str | None:
    """Summarize the conversation for Deep Research. Returns ``None`` on empty input or failure
    (the caller then proceeds without a summary rather than failing the turn)."""
    if not transcript.strip():
        return None

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SUMMARY_SYSTEM_PROMPT), ("user", "{transcript}")]
    )
    llm = get_chat_model(api_key=api_key, model_config=model_config)
    try:
        response = await (prompt | llm).ainvoke({"transcript": transcript})
    except Exception as e:
        logger.warning(f"Deep Research context summarization failed: {e!r}")
        return None

    content = response.content
    summary = content if isinstance(content, str) else str(content)
    return summary.strip() or None


async def triage_questions(
    *,
    api_key: str | SecretStr,
    model_config: LLMModelConfig,
    context: str,
    questions: list[str],
) -> tuple[dict[str, str], list[str]]:
    """Split clarifying questions into ``(answered, pending)``.

    ``answered`` maps a question to the answer derived from context; ``pending`` is the list of
    questions (verbatim, original wording) the user must still answer. On any failure or missing
    context every question is treated as pending, so the user is asked rather than the flow guessing.
    """
    if not questions:
        return {}, []
    if not context.strip():
        return {}, list(questions)

    numbered = "\n".join(f"{i}: {q}" for i, q in enumerate(questions))
    prompt = ChatPromptTemplate.from_messages([("system", _TRIAGE_SYSTEM_PROMPT)])
    llm = get_chat_model(api_key=api_key, model_config=model_config).with_structured_output(
        _TriageResponse, method="json_mode"
    )
    try:
        result = await (prompt | llm).ainvoke({"context": context, "questions": numbered})
        assert isinstance(result, _TriageResponse)
    except Exception as e:
        logger.warning(f"Deep Research question triage failed; asking the user everything: {e!r}")
        return {}, list(questions)

    decisions = {item.index: item for item in result.items}
    answered: dict[str, str] = {}
    pending: list[str] = []
    for idx, question in enumerate(questions):
        decision = decisions.get(idx)
        if decision is not None and decision.can_answer and decision.answer:
            answered[question] = decision.answer.strip()
        else:
            # Keep the original wording so the user sees Deep Research's question verbatim.
            pending.append(question)
    return answered, pending


def format_auto_answers(answered: dict[str, str]) -> list[str]:
    """Format answered questions as ``"Q: ...\\nA: ..."`` blocks for replay to Deep Research."""
    return [f"Q: {question}\nA: {answer}" for question, answer in answered.items()]

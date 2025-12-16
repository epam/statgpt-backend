import os

from langchain_core.prompts import ChatPromptTemplate

from statgpt.common.utils.files import read_yaml


class HybridIndexerDefaultPrompts:
    NORMALIZE_SYSTEM_PROMPT: str
    NORMALIZE_USER_PROMPT: str

    HARMONIZE_SYSTEM_PROMPT: str
    HARMONIZE_USER_PROMPT: str

    @classmethod
    def init_from_config(cls, config: dict[str, dict[str, str]]) -> None:
        cls.NORMALIZE_SYSTEM_PROMPT = config["normalize"]["systemPrompt"]
        cls.NORMALIZE_USER_PROMPT = config["normalize"]["userPrompt"]

        cls.HARMONIZE_SYSTEM_PROMPT = config["harmonize"]["systemPrompt"]
        cls.HARMONIZE_USER_PROMPT = config["harmonize"]["userPrompt"]

    @classmethod
    def get_normalize_prompts(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", cls.NORMALIZE_SYSTEM_PROMPT),
                ("human", cls.NORMALIZE_USER_PROMPT),
            ]
        )

    @classmethod
    def get_harmonize_prompts(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", cls.HARMONIZE_SYSTEM_PROMPT),
                ("human", cls.HARMONIZE_USER_PROMPT),
            ]
        )


def load_prompts():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    fp = os.path.join(script_dir, "assets", "hybrid_indexer.yaml")
    prompts_raw = read_yaml(fp)
    HybridIndexerDefaultPrompts.init_from_config(prompts_raw)


load_prompts()

import os

from langchain_core.prompts import ChatPromptTemplate

from common.utils.files import read_yaml


class IndexerPrompts:
    NORMALIZE_SYSTEM_PROMPT: str
    NORMALIZE_USER_PROMPT: str

    HARMONIZE_SYSTEM_PROMPT: str
    HARMONIZE_USER_PROMPT: str

    SEARCH_NORMALIZE_SYSTEM_PROMPT: str
    SEARCH_NORMALIZE_USER_PROMPT: str

    SEPARATE_SUBJECTS_SYSTEM_PROMPT: str
    SEPARATE_SUBJECTS_USER_PROMPT: str

    RELEVANCE_SYSTEM_PROMPT: str
    RELEVANCE_USER_PROMPT: str

    @classmethod
    def init_from_config(cls, config: dict[str, dict[str, str]]) -> None:
        cls.NORMALIZE_SYSTEM_PROMPT = config["normalize"]["systemPrompt"]
        cls.NORMALIZE_USER_PROMPT = config["normalize"]["userPrompt"]

        cls.HARMONIZE_SYSTEM_PROMPT = config["harmonize"]["systemPrompt"]
        cls.HARMONIZE_USER_PROMPT = config["harmonize"]["userPrompt"]

        cls.SEARCH_NORMALIZE_SYSTEM_PROMPT = config["search_normalize"]["systemPrompt"]
        cls.SEARCH_NORMALIZE_USER_PROMPT = config["search_normalize"]["userPrompt"]

        cls.SEPARATE_SUBJECTS_SYSTEM_PROMPT = config["separate_subjects"]["systemPrompt"]
        cls.SEPARATE_SUBJECTS_USER_PROMPT = config["separate_subjects"]["userPrompt"]

        cls.RELEVANCE_SYSTEM_PROMPT = config["relevance"]["systemPrompt"]
        cls.RELEVANCE_USER_PROMPT = config["relevance"]["userPrompt"]

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

    @classmethod
    def get_search_normalize_prompts(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", cls.SEARCH_NORMALIZE_SYSTEM_PROMPT),
                ("human", cls.SEARCH_NORMALIZE_USER_PROMPT),
            ]
        )

    @classmethod
    def get_separate_subjects_prompts(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", cls.SEPARATE_SUBJECTS_SYSTEM_PROMPT),
                ("human", cls.SEPARATE_SUBJECTS_USER_PROMPT),
            ]
        )

    @classmethod
    def get_relevance_prompts(cls) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", cls.RELEVANCE_SYSTEM_PROMPT),
                ("human", cls.RELEVANCE_USER_PROMPT),
            ]
        )


def load_prompts():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = os.path.join(script_dir, "assets")

    indexer_path = os.path.join(config_dir, "indexer.yaml")
    indicator_config = read_yaml(indexer_path)

    IndexerPrompts.init_from_config(indicator_config)


load_prompts()

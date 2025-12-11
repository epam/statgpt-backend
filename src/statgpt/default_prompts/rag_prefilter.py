import os

from common.schemas import DefaltPromptsBase


class RAGPrefilterDefaultPrompts(DefaltPromptsBase):
    date_system_prompt: str
    latest_system_prompt: str
    publication_types_system_prompt: str
    last_n_publications_system_prompt: str


yaml_fp = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "rag_prefilter.yaml")
rag_prefilter_default_prompts = RAGPrefilterDefaultPrompts.from_yaml(fp=yaml_fp)

import os

from statgpt.common.schemas import DefaltPromptsBase


class DatasetsMetadataDefaultPrompts(DefaltPromptsBase):
    system_prompt: str


yaml_fp = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "assets", "datasets_metadata.yaml"
)
datasets_metadata_default_prompts = DatasetsMetadataDefaultPrompts.from_yaml(fp=yaml_fp)

"""Configuration of a channel's discovery datasets.

One block owns both halves of the feature: where the channel's discovery records are published,
and how the published documents are surfaced during a chat turn. Written as a tool config even
though the Supreme Agent never calls it - the chat-time lookup runs inside the data query tool -
so promoting it to a tool of its own later is a matter of registering a `StatGptTool` for it.
"""

from pydantic import AliasChoices, Field, PositiveInt

from statgpt.common.config import utils as config_utils

from .base import BaseYamlModel, SystemUserPrompt
from .model_config import LLMModelConfig
from .tool_details import BaseToolDetails


class DiscoveryDatasetsPrompts(BaseYamlModel):
    relevance_prompt: SystemUserPrompt | None = Field(
        default=None,
        description=(
            "Prompt asking the model which of the retrieved discovery datasets are relevant to"
            " the query. Falls back to the built-in default when unset."
        ),
    )


class DiscoveryDatasetsTemplates(BaseYamlModel):
    """How the datasets the model kept are rendered into the data query response."""

    wrapper: str = Field(
        description=(
            "Rendered once around the joined items; must contain the `{items}` placeholder."
            " Emitted only when at least one dataset was found relevant."
        )
    )
    item: str = Field(
        description=(
            "Rendered once per relevant dataset. Placeholders are the document's metadata"
            " fields plus `description`, `display_name`, `document_id`, `rank` and `reason`;"
            " an unknown placeholder renders as empty."
        )
    )


class DiscoveryDatasetsDetails(BaseToolDetails):
    application_id_raw: str = Field(
        validation_alias=AliasChoices("application_id", "applicationId"),
        serialization_alias="applicationId",
        description=(
            "The DIAL application id of the Generic RAG channel holding this channel's"
            " discovery records. Supports $env:{VAR} syntax."
        ),
    )
    top_n: PositiveInt = Field(
        default=32,
        description=(
            "How many documents to retrieve. The service applies this both per index and to"
            " the rank-fused result, so it is a ceiling rather than a target count."
        ),
    )
    indexes: list[str] | None = Field(
        default=None,
        description=(
            "Which document indexes to search, in the order the stages run. Leave unset to let"
            " the RAG channel use every index it flags as part of its hybrid search."
        ),
    )
    llm_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    prompts: DiscoveryDatasetsPrompts = Field(default_factory=DiscoveryDatasetsPrompts)  # type: ignore[assignment]
    templates: DiscoveryDatasetsTemplates
    debug_stage_name: str = Field(
        default="Discovery Datasets Lookup",
        description=(
            "Name of the debug stage carrying the retrieved documents, the model's verdicts and"
            " any error. Always debug-only: the lookup has no user-facing progress to report."
        ),
    )

    def get_application_id(self) -> str:
        return config_utils.replace_env(self.application_id_raw)

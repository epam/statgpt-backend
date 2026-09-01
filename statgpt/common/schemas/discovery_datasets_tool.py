"""Configuration of a channel's discovery datasets.

One block owns both halves of the feature: where the channel's discovery records are published,
and how the published documents are surfaced during a chat turn.
"""

from enum import StrEnum

from pydantic import AliasChoices, Field, PositiveInt

from statgpt.common.config import utils as config_utils

from .base import BaseYamlModel, SystemUserPrompt
from .model_config import LLMModelConfig
from .tool_details import BaseToolDetails


class DiscoveryPreFilterAxis(StrEnum):
    """A dimension the discovery search's candidate set can be narrowed on."""

    REFERENCE_AREA = "reference_area"
    PARTNER_REFERENCE_AREA = "partner_reference_area"
    FREQUENCY = "frequency"
    AGENCY = "agency"


class DiscoveryDatasetsPrompts(BaseYamlModel):
    relevance_prompt: SystemUserPrompt | None = Field(
        default=None,
        description=(
            "Prompt asking the model which of the retrieved discovery datasets are relevant to"
            " the query. Falls back to the built-in default when unset."
        ),
    )


class DiscoveryPreFilterPrompts(BaseYamlModel):
    """One prompt per pre-filter axis. Each falls back to the built-in default when unset.

    Every prompt is given `{query}` and `{values}`, the values being the vocabulary that axis
    may select from - a prompt that ignores `{values}` invites selections that are then dropped.
    """

    reference_area_prompt: SystemUserPrompt | None = None
    partner_reference_area_prompt: SystemUserPrompt | None = None
    frequency_prompt: SystemUserPrompt | None = None
    agency_prompt: SystemUserPrompt | None = None


class DiscoveryPreFilterConfig(BaseYamlModel):
    """How the discovery search's candidate set is narrowed before it is ranked.

    Every setting here only ever removes candidates, so switching the whole block off - or
    losing any part of it at runtime - falls back to searching the channel's own documents,
    which is what the lookup did before.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Whether the query is read for the values it can be narrowed to before the"
            " discovery search. Disabled means the search sees every document this channel"
            " published, as it did before pre-filtering existed."
        ),
    )
    axes: list[DiscoveryPreFilterAxis] = Field(
        default_factory=lambda: list(DiscoveryPreFilterAxis),
        description=(
            "Which dimensions the query may be narrowed on. An axis nothing was selected for,"
            " or whose vocabulary could not be read, drops out and leaves the others standing."
            " The two area axes are alternatives to each other rather than a further dimension:"
            " a record matches by covering an area as its subject or as a partner."
        ),
    )
    reference_area_top_n: PositiveInt = Field(
        default=20,
        description=(
            "How many reference-area candidates the vocabulary search offers the model, per area"
            " axis. Bounds both the prompt and, with the other axes, the number of filters a"
            " search carries."
        ),
    )
    llm_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    prompts: DiscoveryPreFilterPrompts = Field(default_factory=DiscoveryPreFilterPrompts)  # type: ignore[assignment]


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
    reference_area_application_id_raw: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "reference_area_application_id", "referenceAreaApplicationId"
        ),
        serialization_alias="referenceAreaApplicationId",
        description=(
            "The DIAL application id of the Generic RAG channel holding this channel's"
            " reference-area vocabulary, which the indexing job publishes and the pre-filter"
            " resolves a query's areas against. Unset means no vocabulary is published and the"
            " reference-area axis is unavailable. Supports $env:{VAR} syntax."
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
    pre_filter: DiscoveryPreFilterConfig = Field(default_factory=DiscoveryPreFilterConfig)  # type: ignore[assignment]
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

    def get_reference_area_application_id(self) -> str | None:
        if self.reference_area_application_id_raw is None:
            return None
        return config_utils.replace_env(self.reference_area_application_id_raw)

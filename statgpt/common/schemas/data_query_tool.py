from typing import Any, Self

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import FieldValidationInfo

from statgpt.common.config import LLMModelsEnum
from statgpt.common.config.utils import replace_env

from .base import BaseYamlModel, SystemUserPrompt
from .enums import (
    IndexerVersion,
    IndicatorSelectionVersion,
    SpecialDimensionsProcessorType,
    TimePeriodStrategy,
)
from .model_config import LLMModelConfig
from .tool_details import BaseToolDetails, StageDescriptor


def bool_from_str(value: str) -> bool:
    """
    Converts a string to a boolean value.
    If the string is an environment variable reference, it will be replaced with its value before conversion.
    """
    return TypeAdapter(bool).validate_python(replace_env(value))


class DataQueryPrompts(BaseYamlModel):
    datetime_prompt: str | None = Field(default=None)
    group_expander_prompt: str | None = Field(default=None)
    group_expander_fallback_prompt: str | None = Field(default=None)
    normalization_prompt: str | None = Field(default=None)
    named_entities_prompt: str | None = Field(default=None)
    indicators_selection_system_prompt: str | None = Field(default=None)
    validation_system_prompt: str | None = Field(default=None)
    validation_user_prompt: str | None = Field(default=None)
    dataset_selection_prompt: SystemUserPrompt | None = Field(default=None)
    incomplete_queries_prompt: str | None = Field(default=None)
    summarize_queries_prompt: str | None = Field(default=None)


class DataQueryMessages(BaseYamlModel):
    no_data_for_country: str | None = Field(
        default=None,
        description="Message for the no data for country response, can contain {country_details} placeholder",
    )
    no_data: str | None = Field(
        default=None,
        description="Message for the no data response",
    )
    data_query_executed_agent_only: str | None = Field(
        default=None,
        description="Message for the data query executed response, only for agent, won't be shown to the user. "
        "Will be appended to the end of the tool response if present.",
    )
    multiple_datasets_agent_only: str | None = Field(
        default=None,
        description="Message for the multiple datasets response, only for agent, won't be shown to the user. "
        "Will be appended to the end of the tool response if present.",
    )
    invalid_time_period: str | None = Field(
        default=None,
        description="Message when all built queries have time periods that are out of range.",
    )


class ToolAttachment(BaseYamlModel):
    enabled_str: str = Field(
        description=(
            "Whether the tool should return this attachment."
            " The value can be a reference to an environment variable."
        )
    )
    name: str | None = Field(
        default=None, description="Attachment name template, may be None if disabled."
    )

    @field_validator('enabled_str', mode='after')
    @classmethod
    def validate_enabled(cls, enabled: str) -> str:
        """Validate the `enabled` field to ensure it can return a boolean value."""
        try:
            bool_from_str(enabled)
        except Exception as e:
            raise ValueError(f"Invalid value for enabled_str: {enabled}. Error: {e}")
        return enabled

    @field_validator("name", mode="after")
    def validate_name(cls, name: str | None, info: FieldValidationInfo) -> str | None:
        """Validate the `name` field to ensure it is not empty if `enabled` is True."""
        enabled_str = info.data.get("enabled_str")
        if enabled_str and bool_from_str(enabled_str) and not name:
            raise ValueError("Attachment name must be provided if the attachment is enabled.")
        return name

    @property
    def enabled(self) -> bool:
        return bool_from_str(self.enabled_str)


class DataQueryAttachments(BaseYamlModel):
    """Represents the attachments that can be returned by the data query tool."""

    custom_table: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(enabled_str="True", name="Data: {dataset_source_id}")
    )
    plotly_grid: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(
            enabled_str="False", name="Plotly Grid: {dataset_source_id}"
        )
    )
    csv_file: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(
            enabled_str="True", name="Data (CSV): {dataset_source_id}.csv"
        )
    )
    plotly_graphs: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(enabled_str="True", name="Graph: {figure_title}")
    )
    json_query: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(
            enabled_str="False", name="Query (JSON): {dataset_source_id}"
        )
    )
    python_code: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(
            enabled_str="False", name="Python Code: {dataset_source_id}"
        )
    )
    merged_python_code: ToolAttachment = Field(
        default_factory=lambda: ToolAttachment(enabled_str="False", name="Python Code")
    )


class DataQueryLLMModels(BaseYamlModel):
    datasets_selection_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    dimensions_selection_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    indicators_selection_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    incomplete_queries_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    group_expander_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    named_entities_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    time_period_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    query_normalization_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    summarize_queries_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)


class SpecialDimensionsProcessor(BaseYamlModel):
    id: str = Field(description="Unique identifier of the processor.")
    # alias: str = Field(description="Alias for dimensions.")
    type: SpecialDimensionsProcessorType = Field()
    llm_model_config: LLMModelConfig = Field(default_factory=LLMModelConfig)
    # TODO: top_k is specific to LHCL, move it to LHCL config subclass later
    top_k: PositiveInt = Field(
        description="Number of candidates retrieved from vector search", default=50
    )
    prompt: SystemUserPrompt


class HybridSearchPrompts(BaseYamlModel):
    relevancy_prompts: SystemUserPrompt | None = Field(default=None)


class HybridSearchConfig(BaseYamlModel):
    """Configuration for the Hybrid Search and Indexer."""

    # ~~~~~~~~~~ Indexer config ~~~~~~~~~~

    indexer_alpha: float = Field(
        default=0.99,
        description="Weight for semantic score in convex combination with lexical score",
    )
    ignored_term_ids: set[str] = Field(
        default_factory=lambda: {"_Z"}, description="Set of term IDs not to include in series name"
    )

    normalize_model_config: LLMModelConfig = Field(
        description="LLM Model used for normalization",
        default_factory=lambda: LLMModelConfig(deployment=LLMModelsEnum.GPT_4_1_MINI_2025_04_14),
    )
    harmonize_model_config: LLMModelConfig = Field(
        description="LLM Model used for harmonization",
        default_factory=lambda: LLMModelConfig(deployment=LLMModelsEnum.GPT_4_1_MINI_2025_04_14),
    )

    # ~~~~~~~~~~ Search config ~~~~~~~~~~

    search_model_config: LLMModelConfig = Field(
        description="LLM Model used for search",
        default_factory=LLMModelConfig,
    )

    DEFAULT_MAX_CANDIDATES: PositiveInt = 32

    max_candidates: PositiveInt = Field(
        default=DEFAULT_MAX_CANDIDATES,
        description=(
            "The maximum amount of candidates to be returned from hybrid search. "
            "If query is complex and will be split into multiple simple queries by Hybrid Search, "
            "this limit will be applied to each simple query."
        ),
    )
    max_lexical_pre_match_candidates: PositiveInt = Field(default=DEFAULT_MAX_CANDIDATES)
    max_lexical_candidates: PositiveInt = Field(
        default=4 * DEFAULT_MAX_CANDIDATES,
        description="The number of candidates to be searched by lexical search.",
    )
    max_semantic_candidates: PositiveInt = Field(
        default=2 * DEFAULT_MAX_CANDIDATES,
        description="The number of candidates to be searched by semantic search.",
    )
    max_lexical_only_candidates: NonNegativeInt = Field(
        default=8,
        description=(
            "Number of top lexical-only candidates to force into the LLM relevance set. "
            "Fusion is anchored on semantic results, so a strong keyword match that falls "
            "outside the semantic top-k (`max_semantic_candidates`) is otherwise dropped before "
            "the LLM judge and can never be selected — a pure recall loss on rare/coded indicators "
            "with weak embeddings. These candidates are availability-filtered and ranked by lexical "
            "score, and are added on top of the diversified hybrid candidates. "
            "Set to 0 to restore the previous semantic-anchored behavior."
        ),
    )

    default_alpha: float = Field(default=0.9)
    hybrid_alpha: float = Field(default=0.8)
    fallback_alpha: float = Field(
        default=0.999, description="Alpha in case of fallback to semantic search."
    )

    max_output_div: PositiveInt = Field(default=4)
    batch_size: PositiveInt = Field(default=32)
    named_entities_to_remove: list[str] = Field(
        default_factory=list, description="Named entities to remove from the search query."
    )

    disable_separate_subjects: bool = Field(
        default=True,
        description=(
            "Whether to split the normalized query into multiple subject sub-queries via LLM "
            "before searching. When disabled, the normalized query is searched as a single query."
        ),
    )
    include_lower_scored_datasets: bool = Field(
        default=False,
        description=(
            "When true, keep the best-scored indicators of each dataset that passes the score "
            "thresholds, even if other datasets have higher-scored indicators. "
            "When false, keep only the indicators with the best overall score."
        ),
    )
    use_only_best_score: bool | None = Field(
        default=None,
        deprecated=(
            "`use_only_best_score` is deprecated; use `include_lower_scored_datasets` instead "
            "(inverse meaning)."
        ),
        description=(
            "Deprecated. Superseded by `include_lower_scored_datasets`, which has the inverse "
            "meaning (`use_only_best_score=True` == `include_lower_scored_datasets=False`). "
            "Kept for backward compatibility: when `include_lower_scored_datasets` is not set, "
            "this value is used. Prefer `include_lower_scored_datasets` in new configs."
        ),
    )
    single_dataset_score_threshold: int = Field(
        default=2,
        description="Relevance score threshold for when indicators are available only from a single dataset.",
        ge=0,
        le=3,
    )
    multi_dataset_score_threshold: int = Field(
        default=2,
        description="Relevance score threshold for when indicators are available from multiple datasets.",
        ge=0,
        le=3,
    )
    prompts: HybridSearchPrompts = Field(default_factory=HybridSearchPrompts)

    @model_validator(mode="after")
    def _apply_deprecated_use_only_best_score(self) -> Self:
        """Honor the deprecated `use_only_best_score` for backward compatibility.

        `include_lower_scored_datasets` replaced `use_only_best_score` with inverse
        meaning. Old channel configs still set `useOnlyBestScore`, so we map it onto
        the new field when the new one is not explicitly set. When both are set, the
        new field wins.
        """
        # read via __dict__: attribute access on a deprecated field emits a DeprecationWarning
        use_only_best_score: bool | None = self.__dict__.get("use_only_best_score")
        if (
            "include_lower_scored_datasets" not in self.model_fields_set
            and use_only_best_score is not None
        ):
            self.include_lower_scored_datasets = not use_only_best_score
        return self


class DataQueryStageNames(BaseYamlModel):
    """Descriptors for the primary non-debug pipeline stages of the data_query tool.

    In YAML, each stage is written as a plain string giving its display name; the
    field name is the stable logical key used by `StagesConfig.rules[].key` and is
    bound here as a private attribute so it cannot be overridden from YAML and call
    sites never have to type the key as a string literal.

    Example YAML:
        pipelineStageNames:
          normalizingQuery: "Preparing your query"
    """

    normalizing_query: StageDescriptor = Field(
        default_factory=lambda: StageDescriptor(name="Normalizing Query")
    )
    extracting_named_entities: StageDescriptor = Field(
        default_factory=lambda: StageDescriptor(name="Extracting Named Entities")
    )
    hybrid_indicators_selection: StageDescriptor = Field(
        default_factory=lambda: StageDescriptor(name="Hybrid Indicators Selection")
    )
    selecting_indicators: StageDescriptor = Field(
        default_factory=lambda: StageDescriptor(name="Selecting Indicators")
    )
    selecting_special_dimensions: StageDescriptor = Field(
        default_factory=lambda: StageDescriptor(name="Selecting Special Dimensions")
    )
    constructing_data_query: StageDescriptor = Field(
        default_factory=lambda: StageDescriptor(name="Constructing Data Query")
    )
    executing_data_query: StageDescriptor = Field(
        default_factory=lambda: StageDescriptor(name="Executing Data Query")
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_strings(cls, data: Any) -> Any:
        """Accept a plain string per field as shorthand for {"name": <string>}.

        Handles both the snake_case field name and its camelCase alias, since
        `BaseYamlModel` enables `populate_by_name=True` and channel YAMLs are
        written in camelCase.
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for field_name, field_info in cls.model_fields.items():
            for key in (field_name, field_info.alias):
                if key is None:
                    continue
                value = out.get(key)
                if isinstance(value, str):
                    out[key] = {"name": value}
        return out

    def model_post_init(self, __context: Any) -> None:
        """Bind each descriptor's `_key` private attribute to its field name."""
        for field_name in self.__class__.model_fields:
            descriptor: StageDescriptor = getattr(self, field_name)
            descriptor._key = field_name


class DataQueryDetails(BaseToolDetails):
    indexer_version: IndexerVersion = Field(
        default=IndexerVersion.semantic, description="The version of the indexer"
    )
    indicator_selection_version: IndicatorSelectionVersion = Field(
        default=IndicatorSelectionVersion.semantic_v4,
        description="The version of the indicator selection algorithm",
    )
    candidates_per_entity: PositiveInt = Field(
        default=30,
        description="The maximum number of non-indicator dimension candidates to retrieve per named entity",
    )
    hybrid_search_config: HybridSearchConfig | None = Field(default=None)
    special_dimensions_processors: list[SpecialDimensionsProcessor] = Field(default_factory=list)
    filter_by_country_entities: bool = Field(
        default=True,
        description="Whether to filter dataset queries by presence of 'country' named entities",
    )
    clarify_if_multiple_datasets: bool = Field(
        default=True,
        description=(
            "Whether to ask the user for clarification if multiple datasets are "
            "found to match the query."
        ),
    )
    time_period_strategy: TimePeriodStrategy = Field(
        default=TimePeriodStrategy.BEFORE,
        description="A strategy that determines when to apply time periods for filtering in availability queries.",
    )
    llm_models: DataQueryLLMModels = Field(default_factory=DataQueryLLMModels)  # type: ignore
    prompts: DataQueryPrompts = Field(default_factory=DataQueryPrompts)  # type: ignore
    messages: DataQueryMessages = Field(default_factory=DataQueryMessages)  # type: ignore
    attachments: DataQueryAttachments = Field(default_factory=DataQueryAttachments)  # type: ignore
    pipeline_stage_names: DataQueryStageNames = Field(default_factory=DataQueryStageNames)
    allow_auto_update: bool = Field(
        default=False,
        description="Whether datasets in this channel should be auto-updated by the batch auto-update script.",
    )
    tool_response_max_cells: PositiveInt = Field(
        default=300,
        description=(
            "Maximum number of cells to include in the tool response. If the result exceeds this number, "
            "the data won't be included in the response shown to agent. The user will always see the data in the "
            "UI table, regardless of this limitation."
        ),
        ge=0,
    )

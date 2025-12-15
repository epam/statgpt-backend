from pydantic import BaseModel, Field

from statgpt.app.schemas.query_builder import (
    DatasetAvailabilityQueriesType,
    DateTimeQueryResponse,
    LLMSelectionDimensionCandidate,
    NamedEntitiesResponse,
    NamedEntity,
    RetrievalStagesResults,
    SelectedCandidates,
    SpecialDimensionChainOutput,
)
from statgpt.app.services.chat_facade import (
    ChannelServiceFacade,
    ScoredDimensionCandidate,
    VersionedDataSet,
)
from statgpt.common.auth.auth_context import AuthContext


class SearchInput(BaseModel):
    """
    Input parameters for dimension search chains.
    Contains only fields required by the dimension search components.
    """

    # Core service dependencies
    auth_context: AuthContext = Field(description="Authentication context for data access")
    data_service: ChannelServiceFacade = Field(
        description="Service facade for accessing data channels"
    )
    datasets_dict: dict[str, VersionedDataSet] = Field(
        default_factory=dict, description="Selected datasets for the query"
    )

    # Query state - shared across all searches
    strong_queries: DatasetAvailabilityQueriesType = Field(
        default_factory=dict, description="Strong (validated) queries to datasets"
    )
    strong_queries_best_nonempty_attempt: DatasetAvailabilityQueriesType = Field(
        default_factory=dict,
        description="Best non-empty attempt at building strong queries (fallback)",
    )
    strong_availability: DatasetAvailabilityQueriesType = Field(
        default_factory=dict, description="Availability results for strong queries"
    )

    # Non-indicator specific fields
    strong_queries_nonindicators: DatasetAvailabilityQueriesType = Field(
        default_factory=dict, description="Strong queries for non-indicator dimensions only"
    )
    weak_queries_nonindicators: DatasetAvailabilityQueriesType = Field(
        default_factory=dict, description="Weak queries for non-indicator dimensions"
    )
    named_entities_response: NamedEntitiesResponse = Field(
        default_factory=NamedEntitiesResponse, description="Named entities detected in user query"
    )
    country_named_entities: list[NamedEntity] = Field(
        default_factory=list, description="Country named entities for filtering"
    )
    dimension_candidates: list[ScoredDimensionCandidate] = Field(
        default_factory=list, description="Dimension candidates from vector search"
    )
    dimension_candidates_for_llm_selection: list[LLMSelectionDimensionCandidate] = Field(
        default_factory=list, description="Dimension candidates prepared for LLM selection"
    )
    dimension_values_llm_selection_output: SelectedCandidates | None = Field(
        default=None, description="LLM selection output for dimension values"
    )
    date_time_query_response: DateTimeQueryResponse = Field(
        default_factory=DateTimeQueryResponse, description="Parsed date/time query information"
    )

    # Indicator specific fields
    retrieval_results: RetrievalStagesResults = Field(
        default_factory=RetrievalStagesResults,
        description="Retrieval results from indicator selection stages",
    )

    # Special dimensions fields
    special_dims_outputs: dict[str, SpecialDimensionChainOutput] = Field(
        default_factory=dict,
        description="Mapping from SpecialDimensionsProcessor.id to its chain output",
    )

    model_config = {"arbitrary_types_allowed": True}

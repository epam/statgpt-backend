import typing as t

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.base import (
    DataSet,
    DataSetAvailabilityQuery,
    DataSetQuery,
    DimensionQuery,
    QueryOperator,
)
from common.data.sdmx.v21.dataset import Sdmx21DataSet
from common.schemas import ToolTypes
from statgpt.services import (
    ChannelServiceFacade,
    ScoredDimensionCandidate,
    ScoredIndicatorCandidate,
)

from .selection_candidates import (
    BatchedSelectionOutputBase,
    LLMSelectionCandidateBase,
    SelectedCandidates,
)
from .tool_states import ToolMessageState

# custom types:
# dataset -> dimension -> dimension values
DatasetDimQueriesType: t.TypeAlias = dict[str, dict[str, list[str]]]
# dataset_id -> dimesnsion_id -> term_id -> term name
# Example: {"3": {"COUNTRY": {"111": "United States"}}}
DatasetDimensionTermNameType: t.TypeAlias = dict[str, dict[str, dict[str, str]]]
# dataset_id -> availability query
DatasetAvailabilityQueriesType: t.TypeAlias = dict[str, DataSetAvailabilityQuery]


class DatasetDimQueries(BaseModel):
    """
    We use pydantic model on top of the custom type to:
    1. perform data validations
    2. implement "is_valid()" method
    """

    queries: DatasetDimQueriesType = Field(
        default={},
        description=(
            "Mapping from dataset id to mapping from dimension id to list of dimension values"
        ),
    )

    def is_valid(self) -> bool:
        """
        Check if there is at least 1 dataset query
        with at least 1 dimension query
        with at least 1 value.
        """
        for dim_queries in self.queries.values():
            if any(dim_values for dim_values in dim_queries.values()):
                return True
        return False


class DateTimeQueryResponse(BaseModel):
    # NOTE: we use default=False value for "forecast" and "time_period_specified" fields
    # to allow DataQuery state to have empty DateTimeQueryResponse

    forecast: bool = Field(
        default=False, description="Whether time period is related to a forecast"
    )
    start: str | None = Field(default=None, description="The start date formatted as 'YYYY-MM-DD'")
    end: str | None = Field(default=None, description="The end date formatted as 'YYYY-MM-DD'")
    time_period_specified: bool = Field(
        default=False,
        description="Whether user specified time period in their query. "
        "If it isn't, default time period will be applied.",
    )

    def to_query(self) -> DimensionQuery | None:
        if self.start and self.end:
            return DimensionQuery(
                dimension_id="TIME_PERIOD",
                values=[self.start, self.end],
                operator=QueryOperator.BETWEEN,
            )
        elif self.start:
            return DimensionQuery(
                dimension_id="TIME_PERIOD",
                values=[self.start],
                operator=QueryOperator.GREATER_THAN_OR_EQUALS,
            )
        elif self.end:
            return DimensionQuery(
                dimension_id="TIME_PERIOD",
                values=[self.end],
                operator=QueryOperator.LESS_THAN_OR_EQUALS,
            )
        return None

    @model_validator(mode='after')
    def _check_hallucinations_in_time_period_specified(self):
        if not self.time_period_specified:
            if self.forecast or self.start is not None or self.end is not None:
                logger.warning(
                    'HALLUCINATION! LLM incorrectly did not set "time_period_specified" to True: '
                    f'{self}. overriding to True'
                )
                self.time_period_specified = True
        return self


class NamedEntity(BaseModel):
    entity: str = Field(description="The named entity")
    entity_type: str = Field(description="The type of the named entity")

    def to_query(self) -> str:
        return f"{self.entity}; {self.entity_type}"


class NamedEntitiesResponse(BaseModel):
    entities: list[NamedEntity] = Field(
        description="The named entities in the query", default_factory=list
    )


class DataSetsSelectionResponse(BaseModel):
    dataset_ids: list[str] = Field(
        description=(
            "Selected dataset ids. If there is no EXPLICIT datasets specification "
            "in the query, use an empty list."
        ),
        default_factory=list,
    )
    rewritten_query: str = Field(
        default='', description='User query with all detected dataset references removed'
    )


class RetrievalStageDescription(BaseModel):
    field_name: str
    short_name: str
    description: str


class RetrievalStagesResults(BaseModel):
    indicators: dict[str, list[dict]] = Field(
        description="Dictionary mapping stages to their respective list of indicators.",
        default_factory=dict,
    )
    stages_descriptions_ordered: list[RetrievalStageDescription] = Field(
        description="Ordered list of retrieval stages descriptions",
        default_factory=list,
    )


class IndicatorsSearchResult(BaseModel):
    queries: DatasetAvailabilityQueriesType
    retrieval_results: RetrievalStagesResults


class SpecialDimensionChainOutput(BaseModel):
    processor_id: str = Field(description="SpecialDimensionsProcessor.id")
    processor_type: str = Field(description="SpecialDimensionsProcessor.type")
    dataset_queries: dict[str, DimensionQuery] = Field(
        description="mapping from dataset id to special dimension query"
    )
    llm_response: SelectedCandidates

    def no_queries(self):
        """query wasn't built for any dataset"""
        return not (self.dataset_queries) or all(
            q.is_empty() for q in self.dataset_queries.values()
        )


class QueryBuilderAgentState(ToolMessageState):
    """
    Output model to access selected artifacts of a Query Builder Agent.
    """

    type: ToolTypes = ToolTypes.DATA_QUERY

    query: str = Field(default='', description="query from the Data Query tool input")
    query_with_expanded_groups: str = Field(
        default="", description="User query with expanded groups (for now, country groups only)"
    )
    normalized_query_raw: str = Field(
        default="", description="Summarized conversation, before datasets are removed from summary"
    )
    datasets_selection_response: DataSetsSelectionResponse = Field(
        description="LLM response containing ids of selected datasets",
        default_factory=DataSetsSelectionResponse,
    )
    normalized_query: str = Field(default="", description="Summarized conversation")
    date_time_query_response: DateTimeQueryResponse = Field(
        description="The response for the date time query", default_factory=DateTimeQueryResponse
    )
    named_entities_response: NamedEntitiesResponse = Field(
        description="The named entities in the query", default_factory=NamedEntitiesResponse
    )
    indexed_datasets_id_map: dict[str, str] = Field(
        description='Maps dataset entity id to source id, for all datasets indexed by statgpt',
        default_factory=dict,
    )
    strong_queries_nonindicators: DatasetAvailabilityQueriesType = Field(
        description="Strong (filtered by LLM) nonindicator queries",
        default_factory=dict,
    )
    weak_queries: DatasetAvailabilityQueriesType = Field(
        description="Weak (not filtered by LLM) queries to the datasets", default_factory=dict
    )
    strong_queries: DatasetAvailabilityQueriesType = Field(
        description="Strong (filtered by LLM) queries to the datasets", default_factory=dict
    )
    dataset_queries: dict[str, DataSetQuery] = Field(
        description="Queries to the datasets (ready to be sent to source)", default_factory=dict
    )
    retrieval_results: RetrievalStagesResults = Field(
        description='Retrieval stages results, used in evaluations',
        default_factory=RetrievalStagesResults,
    )
    dimension_id_to_name: DatasetDimensionTermNameType = Field(
        description="For dataset queries, contains mapping of datasets id "
        "to their dimension ids to term ids to term names",
        default_factory=dict,
    )
    special_dims_outputs: dict[str, SpecialDimensionChainOutput] = Field(
        description="mapping from SpecialDimensionsProcessor.id to its chain output",
        default_factory=dict,
    )


class LLMSelectionDimensionCandidate(LLMSelectionCandidateBase, ScoredDimensionCandidate):

    # TODO: can separate dedup-propagate logic from concrete dimensions formatting

    index: int
    dedup_key: t.ClassVar[tuple[str, ...]] = ('dimension', 'name')

    @property
    def _id(self) -> str:
        return str(self.index)

    def to_df_row_dict(self) -> dict:
        res = {
            'id': self._id.strip(),
            # TODO: experiment with passing concept name instead of dimension name
            'dimension': self.dimension_alias_or_name.strip(),
            # NOTE: use 'system_code' to avoid LLM confusing with 'id'
            'system_code': self.query_id.strip(),
            'name': self.name.strip(),
            'score': self.score,  # used to sort candidates
        }
        return res

    @staticmethod
    def _candidates_to_df(candidates: list["LLMSelectionDimensionCandidate"]):
        df = pd.DataFrame([c.to_df_row_dict() for c in candidates])
        return df

    @staticmethod
    def _format_row(row: pd.Series) -> str:
        # NOTE: explicitly naming properties (id, system_code)
        # should help to prevent LLM from selecting system code instead of id.
        return f'- {row["name"]} (id: {row["id"]}, system code: {row["system_code"]})'

    @classmethod
    def _format_df(cls, _df: pd.DataFrame) -> str:
        return '\n'.join(_df.apply(cls._format_row, axis=1))

    @classmethod
    def candidates_to_llm_string(cls, candidates: list["LLMSelectionDimensionCandidate"]) -> str:
        """
        Convert candidates to string to ingest to LLM prompt.
        NOTE: we drop duplicates to make it easier for LLM to select ALL relevant items. later we'll propagate selection status to dropped duplicates.
        """
        df = cls._candidates_to_df(candidates)
        df.drop_duplicates(cls.dedup_key, inplace=True)
        df.sort_values('score', ascending=False, inplace=True)

        lines = []
        grouped = df.groupby('dimension', sort=False)
        for ix, (dim_name, df_group) in enumerate(grouped):
            lines.append(f'## dimension: "{dim_name}"\n')
            lines.append(cls._format_df(df_group))
            if ix < len(grouped) - 1:
                lines.append('\n\n')
        text = ''.join(lines)

        return text

    @classmethod
    def propagate_selection_status_to_duplicates(
        cls, candidates: list["LLMSelectionDimensionCandidate"], selected_ids: set[str]
    ) -> list[str]:
        """
        Return ids of candidates selected by LLM, including dropped duplicates.

        Previously we dropped duplicates in the candidates list while preparing LLM string.
        Now we need to propagate selection status to the dropped duplicates.
        """
        if not candidates:
            logger.info('No candidates to propagate selection status to.')
            return []

        df = cls._candidates_to_df(candidates)
        selected_set = set(selected_ids)

        df['selected'] = 0
        df.loc[df['id'].isin(selected_set), 'selected'] = 1
        df['selected_expanded'] = df.groupby(list(cls.dedup_key))['selected'].transform('max')

        selected_expanded_ids = df.loc[df['selected_expanded'] == 1, 'id'].tolist()
        return selected_expanded_ids

    @classmethod
    def from_scored_dimension_candidate(cls, candidate: ScoredDimensionCandidate, index: int):
        return cls(
            index=index,
            score=candidate.score,
            dataset_id=candidate.dataset_id,
            dimension_category=candidate.dimension_category,
        )

    def to_scored_dimension_candidate(self):
        return ScoredDimensionCandidate(
            score=self.score,
            dataset_id=self.dataset_id,
            dimension_category=self.dimension_category,
        )


class ChainState(BaseModel):
    """
    Abstraction over the chain state dictionary.
    Contains fields assigned and used in the chain methods.
    Goal is to provide ease of access and type hinting for the chain state dictionary.
    NOTE: it's not the same as DIAL state assigned to the output message's custom content.
    """

    auth_context: AuthContext
    data_service: ChannelServiceFacade
    query_with_expanded_groups: str = ''
    normalized_query_raw: str = Field(
        default="", description="Summarized conversation, before datasets are removed from summary"
    )
    datasets_selection_response: DataSetsSelectionResponse = DataSetsSelectionResponse()
    normalized_query: str = Field(default="", description="Summarized conversation")
    date_time_query_response: DateTimeQueryResponse = DateTimeQueryResponse()
    # all datasets indexed by statgpt
    datasets_dict_indexed: dict[str, DataSet] = {}
    # selected datasets
    datasets_dict: dict[str, DataSet | Sdmx21DataSet] = {}
    named_entities_response: NamedEntitiesResponse = NamedEntitiesResponse()
    country_named_entities: list[NamedEntity] = []
    dimension_candidates: list[ScoredDimensionCandidate] = []

    weak_queries_nonindicators: DatasetAvailabilityQueriesType = {}
    strong_queries_nonindicators: DatasetAvailabilityQueriesType = {}
    dimension_candidates_for_llm_selection: list[LLMSelectionDimensionCandidate] = []
    dimension_values_llm_selection_output: SelectedCandidates = None  # type: ignore

    weak_queries: DatasetAvailabilityQueriesType = {}
    strong_queries: DatasetAvailabilityQueriesType = {}
    strong_queries_best_nonempty_attempt: DatasetAvailabilityQueriesType = {}
    strong_availability: DatasetAvailabilityQueriesType = {}

    retrieval_results: RetrievalStagesResults = RetrievalStagesResults()
    special_dims_outputs: dict[str, SpecialDimensionChainOutput] = {}
    dataset_queries: dict[str, DataSetQuery] = {}  # final data queries

    indicators_llm_selection_output: BatchedSelectionOutputBase = None  # type: ignore

    # indicator candidates from different retrieval stages.
    # general container - is constantly updated
    indicator_candidates: list[ScoredIndicatorCandidate] = []
    # snapshots of indicator_candidates at different stages.
    indicator_candidates_vector_search_outputs: list[ScoredIndicatorCandidate] = []
    indicator_candidates_filtered_by_availabiility: list[ScoredIndicatorCandidate] = []
    indicator_candidates_llm_selects: list[ScoredIndicatorCandidate] = []

    dimension_id_to_name: DatasetDimensionTermNameType = {}
    dataset_queries_formatted_str: str = ''

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MetaStateKeys:
    """
    used to combine states of multiple data query subchains.
    make 1 subchains return main state that is going to be updated by outputs from other subchains.
    """

    CHAIN_STATE = 'chain_state'  # main state
    SPECIAL_DIMENSIONS_OUTPUTS = 'special_dims_outputs'


class DataSetChoice(BaseModel):
    """
    Represent a dataset choice available for selection by either agent or user.
    """

    id: str = Field(description="The unique identifier of the dataset, used for selection.")
    name: str = Field(description="The human-readable name of the dataset, used for display.")
    description: str | None = Field(
        default=None,
        description="A brief description of the dataset, providing context and details.",
    )
    is_official: bool = Field(
        default=False,
        description="Indicates whether the dataset is official or not.",
    )

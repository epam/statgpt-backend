from datetime import datetime

from aidial_sdk.chat_completion import Choice
from pydantic import BaseModel, ConfigDict, Field

from common.auth.auth_context import AuthContext
from statgpt.services import ChannelServiceFacade


class ChainParametersConfig:
    """Keys to access artifacts shared between different chains/agents."""

    STATE = "state"
    REQUEST = "request"
    CHOICE = "choice"
    AUTH_CONTEXT = "auth_context"
    DATA_SERVICE = "data_service"
    QUERY = "query"
    SKIP_OUT_OF_SCOPE_CHECK = "skip_out_of_scope_check"
    START_OF_REQUEST = "start_of_request"
    CONFIGURATION = "configuration"

    TARGET_PREFILTER = "target_prefilter"
    INDICATOR_ID_2_DATASET_IDS = "indicator_id_2_dataset_ids"
    DATASETS = "datasets"
    DATASETS_DICT = "datasets_dict"
    DATASET_DIMENSION_QUERIES = "dataset_dimension_queries"
    DATASET_QUERIES_FORMATTED_STR = "dataset_queries_formatted_str"
    DATASET_QUERIES = "dataset_queries"
    DATA_RESPONSES = "data_responses"
    SUMMARIZED_DATA_QUERY = 'summarized_data_query'
    HISTORY = "history"
    TARGET = "target"
    OUT_OF_SCOPE = "out_of_scope"
    OUT_OF_SCOPE_REASONING = "out_of_scope_reasoning"
    PERFORMANCE_STAGE = "performance_stage"


class ChainParameters(BaseModel):
    """artifacts shared between different chains/agents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: dict = Field(description='global app state')
    # NOTE: 'request' is pydantic v1 currently - can't use it here
    # request: Request = Field(description="dial request instance")
    choice: Choice = Field(description="dial choice instance")
    auth_context: AuthContext
    data_service: ChannelServiceFacade
    query: str = Field(description="User query")
    skip_out_of_scope_check: bool = Field(description="Whether to skip out-of-scope check")
    start_of_request: datetime = Field(description="Timestamp of the start of the request")

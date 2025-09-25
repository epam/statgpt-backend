import typing as t
from datetime import datetime

from aidial_sdk.chat_completion import Choice, Request, Stage

from common.auth.auth_context import AuthContext
from common.data.base import DataResponse, DataSet, DataSetQuery, DimensionQuery
from statgpt.config import ChainParametersConfig
from statgpt.schemas.dial_app_configuration import StatGPTConfiguration
from statgpt.schemas.file_rags.dial_rag import RagFilterDial
from statgpt.services import ChannelServiceFacade
from statgpt.utils.message_history import History


class ChainParameters:
    @staticmethod
    def skip_out_of_scope_check(inputs: dict) -> bool:
        return inputs[ChainParametersConfig.SKIP_OUT_OF_SCOPE_CHECK]

    @staticmethod
    def is_out_of_scope(inputs: dict) -> bool | None:
        return inputs[ChainParametersConfig.OUT_OF_SCOPE]

    @staticmethod
    def get_out_of_scope_reasoning(inputs: dict) -> str | None:
        return inputs[ChainParametersConfig.OUT_OF_SCOPE_REASONING]

    @staticmethod
    def get_request(data: dict) -> Request:
        return data[ChainParametersConfig.REQUEST]

    @staticmethod
    def get_configuration(data: dict) -> StatGPTConfiguration:
        return data[ChainParametersConfig.CONFIGURATION]

    @staticmethod
    def get_query(data: dict) -> str:
        return data[ChainParametersConfig.QUERY]

    @staticmethod
    def get_target_prefilter(data: dict) -> RagFilterDial | None:
        return data[ChainParametersConfig.TARGET_PREFILTER]

    @staticmethod
    def get_auth_context(data: dict) -> AuthContext:
        return data[ChainParametersConfig.AUTH_CONTEXT]

    @staticmethod
    def get_choice(data: dict) -> Choice:
        return data[ChainParametersConfig.CHOICE]

    @staticmethod
    def get_target(data: dict) -> Stage:
        return data[ChainParametersConfig.TARGET]

    @staticmethod
    def get_history(data: dict) -> History:
        return data[ChainParametersConfig.HISTORY]

    @staticmethod
    def get_data_service(data: dict) -> ChannelServiceFacade:
        return data[ChainParametersConfig.DATA_SERVICE]

    @staticmethod
    def get_indicator_id_2_dataset_ids(data: dict) -> dict[int, list[str]]:
        return data[ChainParametersConfig.INDICATOR_ID_2_DATASET_IDS]

    @staticmethod
    def get_datasets(data: dict) -> t.Sequence[DataSet]:
        """Flat sequence of all datasets that relevant indicator candidates belong to"""
        return data[ChainParametersConfig.DATASETS]

    @staticmethod
    def get_datasets_dict(data: dict) -> dict[str, DataSet]:
        """Flat sequence of all datasets that relevant indicator candidates belong to"""
        return data[ChainParametersConfig.DATASETS_DICT]

    @staticmethod
    def get_dataset_dimension_queries(data: dict) -> dict[str, list[DimensionQuery]]:
        return data[ChainParametersConfig.DATASET_DIMENSION_QUERIES]

    @staticmethod
    def get_dataset_queries_formatted_str(data: dict) -> str:
        return data[ChainParametersConfig.DATASET_QUERIES_FORMATTED_STR]

    @staticmethod
    def get_dataset_queries(data: dict) -> t.Dict[str, DataSetQuery]:
        return data[ChainParametersConfig.DATASET_QUERIES]

    @staticmethod
    def get_data_responses(data: dict) -> dict[str, DataResponse | None]:
        return data[ChainParametersConfig.DATA_RESPONSES]

    @staticmethod
    def get_performance_stage(data: dict) -> Stage:
        return data[ChainParametersConfig.PERFORMANCE_STAGE]

    @staticmethod
    def get_start_of_request(data: dict) -> datetime:
        return data[ChainParametersConfig.START_OF_REQUEST]

    @staticmethod
    def get_state(data: dict) -> dict[str, t.Any]:
        return data[ChainParametersConfig.STATE]

    @staticmethod
    def get_summarized_data_query(data: dict) -> str:
        return data[ChainParametersConfig.SUMMARIZED_DATA_QUERY]

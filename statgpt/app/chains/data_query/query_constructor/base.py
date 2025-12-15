from abc import ABC, abstractmethod

from statgpt.app.schemas.query_builder import ChainState
from statgpt.common.data.base import DataSet, DataSetAvailabilityQuery, DataSetQuery
from statgpt.common.schemas import DataQueryDetails


class BaseQueryConstructor(ABC):

    def __init__(self, config: DataQueryDetails):
        self._config = config

    @abstractmethod
    async def construct_query(
        self,
        ds_query: DataSetAvailabilityQuery,
        ds_availability_query: DataSetAvailabilityQuery,
        dataset: DataSet,
        chain_state: ChainState,
    ) -> DataSetQuery:
        pass

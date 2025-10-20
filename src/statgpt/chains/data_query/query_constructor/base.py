from abc import ABC, abstractmethod

from common.data.base import DataSet, DataSetAvailabilityQuery, DataSetQuery
from statgpt.schemas.query_builder import ChainState


class BaseQueryConstructor(ABC):

    @abstractmethod
    async def construct_query(
        self,
        ds_query: DataSetAvailabilityQuery,
        ds_availability_query: DataSetAvailabilityQuery,
        dataset: DataSet,
        chain_state: ChainState,
    ) -> DataSetQuery:
        pass

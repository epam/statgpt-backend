import logging

from statgpt.app.schemas.query_builder import ChainState
from statgpt.common.data.base import DataSet, DataSetAvailabilityQuery, DataSetQuery
from statgpt.common.schemas import DataQueryDetails

from .base import BaseQueryConstructor

_log = logging.getLogger(__name__)


class CompositeQueryConstructor(BaseQueryConstructor):
    """
    Composite query constructor that tries multiple query construction strategies in sequence.

    Strategy:
    - Try each constructor in order
    - If query is invalid → return immediately (user needs to provide missing dimensions)
    - If query is valid but availability is empty → fallback to next constructor
    - If query is valid and availability is non-empty → return immediately
    - If all constructors produce empty availability, return the last result
    """

    def __init__(self, config: DataQueryDetails, constructors: list[BaseQueryConstructor]):
        super().__init__(config)
        if not constructors:
            raise ValueError("CompositeQueryConstructor requires at least one constructor")
        self._constructors = constructors

    async def construct_query(
        self,
        ds_query: DataSetAvailabilityQuery,
        ds_availability_query: DataSetAvailabilityQuery,
        dataset: DataSet,
        chain_state: ChainState,
    ) -> DataSetQuery:
        dataset_id = dataset.entity_id
        last_result: DataSetQuery | None = None

        for i, constructor in enumerate(self._constructors):
            constructor_name = constructor.__class__.__name__
            _log.info(
                f'[{dataset_id}] CompositeQueryConstructor: trying constructor {i+1}/{len(self._constructors)}: '
                f'{constructor_name}'
            )
            result = await constructor.construct_query(
                ds_query=ds_query,
                ds_availability_query=ds_availability_query,
                dataset=dataset,
                chain_state=chain_state,
            )
            last_result = result

            if not result.is_valid:
                _log.info(
                    f'[{dataset_id}] Constructor {constructor_name} produced invalid query. '
                    f'Returning immediately without trying remaining constructors.'
                )
                return result

            _log.debug(
                f'[{dataset_id}] Constructor {constructor_name} produced valid query. '
                f'Checking availability...'
            )

            try:
                availability_check_query = DataSetAvailabilityQuery.from_dimension_queries_list(
                    result.dimensions_queries
                )
                availability_result = await dataset.availability_query(
                    availability_check_query, chain_state.auth_context
                )
                is_empty = availability_result.is_empty()
                if is_empty:
                    _log.info(
                        f'[{dataset_id}] Constructor {constructor_name} produced valid query '
                        f'but availability is empty. Trying next constructor...'
                    )
                    continue
                else:
                    _log.info(
                        f'[{dataset_id}] Constructor {constructor_name} produced valid query '
                        f'with non-empty availability. Returning result.'
                    )
                    return result

            except Exception as e:
                _log.warning(
                    f'[{dataset_id}] Failed to check availability for constructor {constructor_name}: {e}. '
                    f'Assuming non-empty and returning result.'
                )
                return result

        _log.info(
            f'[{dataset_id}] All constructors tried. '
            f'Returning last result (may have empty availability).'
        )
        return last_result  # type: ignore

import logging

from common.data.base import (
    DataSet,
    DataSetAvailabilityQuery,
    DataSetQuery,
    DimensionQuery,
    DimensionType,
    Query,
    QueryOperator,
)
from statgpt.schemas.query_builder import ChainState

from .base import BaseQueryConstructor

_log = logging.getLogger(__name__)


class IterativeQueryConstructor(BaseQueryConstructor):
    """
    Iterative query constructor tries to auto-set missing dimension queries
    using default queries or available values one by one checking for availability of query for each step.
    """

    @staticmethod
    def _set_dimension_query_from_default_or_available_values(
        dim_id: str,
        dim_type: DimensionType,
        dataset_id: str,
        default_queries: list[Query] | None,
        default_value_codes: list[str],
        availability: Query | None,
    ) -> tuple[DimensionQuery | None, bool]:
        """
        Try to set dimension query for dimension absent in strong queries.
        Use default queries if available, otherwise use available values.

        Returns:
            tuple[DimensionQuery | None, bool]: (dimension_query, was_set_from_default)
        """

        if default_queries:
            default_query = default_queries[0]
            if default_query.values:
                if dim_type != DimensionType.CATEGORY:
                    return DimensionQuery.from_default_query(default_query, dim_id), True

                if availability is not None and availability.values:
                    available_values = availability.values

                    filtered_defaults = set(default_query.values).intersection(available_values)
                    if filtered_defaults:
                        return (
                            DimensionQuery(
                                values=list(filtered_defaults),
                                operator=default_query.operator,
                                dimension_id=dim_id,
                                is_default=True,
                            ),
                            True,
                        )
                    else:
                        _log.debug(
                            'No default values left after filtering by availability for '
                            f'"{dim_id}" dimension in "{dataset_id}" dataset. '
                            'Will try to auto-set using availability data.'
                        )
                else:
                    _log.debug(
                        f'No available values extracted for "{dim_id}" dimension '
                        f'in "{dataset_id}" dataset. '
                        'Will try to auto-set using availability data.'
                    )
            else:
                _log.debug(
                    f'No default values for "{dim_id}" dimension '
                    f'in "{dataset_id}" dataset. '
                    'Will try to auto-set dimension queries using availability data.'
                )
        else:
            _log.debug(
                f'No default queries for "{dim_id}" dimension in "{dataset_id}" dataset. '
                'Will try to auto-set dimension queries using availability data.'
            )

        if dim_type != DimensionType.CATEGORY:
            _log.debug(
                f'Can\'t auto-set query for "{dim_id}" dimension '
                f'in "{dataset_id}" dataset, since it\'s not a categorical dimension.'
            )
            return None, False

        if availability is None or not availability.values:
            _log.debug(
                f'No available values extracted for "{dim_id}" dimension '
                f'in "{dataset_id}" dataset'
            )
            return None, False

        available_values = availability.values

        available_default_values = set(default_value_codes).intersection(available_values)
        if available_default_values:
            _log.debug(
                f'Auto-setting dimension query for "{dim_id}" dimension '
                f'in "{dataset_id}" dataset to available default values: '
                f'{available_default_values}'
            )
            return (
                DimensionQuery(
                    values=list(available_default_values),
                    operator=QueryOperator.IN,
                    dimension_id=dim_id,
                    is_default=True,
                ),
                True,  # set from default
            )

        # TODO: make k_low and k_high configurable
        k_low = 10
        k_high = 40
        if len(available_values) > k_low:
            _log.debug(
                f'Too many available values ({len(available_values)}) '
                f'for "{dim_id}" dimension without default queries '
                f'in "{dataset_id}" dataset. Can\'t auto-set dimension query. '
                f'Sample values: {available_values[:10]}'
            )

            if len(available_values) > k_high:
                # TODO: do not list all available values in clarification question
                # samples = available_values[:k_high]
                pass

            return None, False

        _log.debug(
            f'Auto-setting dimension query for "{dim_id}" dimension '
            f'in "{dataset_id}" dataset to following '
            f'{len(available_values)} available values: {available_values}'
        )
        return (
            DimensionQuery(
                values=list(available_values),  # shallow copy should be enough
                operator=QueryOperator.ALL,
                dimension_id=dim_id,
                is_default=False,
            ),
            False,  # set from availability, not from default
        )

    async def construct_query(
        self,
        ds_query: DataSetAvailabilityQuery,
        ds_availability_query: DataSetAvailabilityQuery,
        dataset: DataSet,
        chain_state: ChainState,
    ) -> DataSetQuery:
        dataset_id = dataset.entity_id
        ds_default_queries = dataset.config.dimension_default_queries
        ds_dimension_queries: dict[str, DimensionQuery] = {
            d.dimension_id: d for d in ds_query.dimensions_queries
        }
        is_ds_query_valid = True

        # Track current availability - will be updated iteratively
        current_availability = ds_availability_query

        # Reorder dimensions: time dimensions first, then others
        dimensions = list(dataset.dimensions())
        dimensions.sort(key=lambda d: 0 if d.is_time_dimension else 1)

        for dimension in dimensions:
            dim_id = dimension.entity_id
            if dim_id in ds_dimension_queries:
                continue

            default_queries = ds_default_queries.get(dim_id)
            availability = current_availability.dimensions_queries_dict.get(dim_id)

            if dimension.dimension_type == DimensionType.DATETIME:
                dtqr = chain_state.date_time_query_response
                if dtqr.time_period_specified:
                    _log.info(
                        f'there is an empty time period filter in dataset "{dataset_id}". '
                        'LLM detected that user specified time period filter to be empty. '
                        'keeping empty time filter, not setting default'
                    )
                    ds_dimension_queries[dim_id] = DimensionQuery(
                        values=['', ''],
                        operator=QueryOperator.BETWEEN,
                        dimension_id=dim_id,
                        is_default=False,
                    )
                    continue
                else:
                    _log.info(
                        f'there is an empty time period filter in dataset "{dataset_id}". '
                        'LLM detected that user did not specify time period. '
                        f'using default time period: {default_queries}'
                    )

            dim_query, was_set_from_default = (
                self._set_dimension_query_from_default_or_available_values(
                    dim_id=dim_id,
                    dim_type=dimension.dimension_type,
                    dataset_id=dataset.source_id,
                    default_queries=default_queries,
                    default_value_codes=dataset.default_value_codes,
                    availability=availability,
                )
            )

            if dim_query is not None:
                ds_dimension_queries[dim_id] = dim_query

                # Run availability query after setting a dimension value in two cases:
                # 1. If we set this dimension from default values
                # 2. If this is a time dimension (regardless of source)
                should_run_availability = was_set_from_default or dimension.is_time_dimension

                if should_run_availability:
                    reason = (
                        "set from default" if was_set_from_default else "time dimension was set"
                    )
                    _log.info(
                        f'Dimension "{dim_id}" was {reason}. '
                        f'Running availability query to update remaining dimensions.'
                    )
                    current_query = DataSetAvailabilityQuery.from_dimension_queries_list(
                        list(ds_dimension_queries.values())
                    )
                    try:
                        current_availability = await dataset.availability_query(
                            current_query, chain_state.auth_context
                        )
                        _log.debug(
                            f'Updated availability after adding "{dim_id}": '
                            f'{current_availability.dimensions_queries_dict.keys()}'
                        )
                    except Exception as e:
                        _log.warning(
                            f'Failed to run availability query after adding "{dim_id}": {e}. '
                            f'Continuing with previous availability data.'
                        )
            else:
                # Dimension query is missing, marking query as invalid.
                # Don't break - need to detect ALL missing dimensions.
                is_ds_query_valid = False

        return DataSetQuery(
            dimensions_queries=list(ds_dimension_queries.values()),
            is_valid=is_ds_query_valid,
        )

import logging
from datetime import date, timedelta
from enum import StrEnum

from common.data.base import DataSet, DataSetQuery, DimensionQuery, QueryOperator
from common.utils.timer import debug_timer
from statgpt.schemas.query_builder import ChainState

_log = logging.getLogger(__name__)


class FrequencyEnum(StrEnum):
    ANNUAL = 'A'
    QUARTERLY = 'Q'
    MONTHLY = 'M'
    WEEKLY = 'W'
    DAILY = 'D'

    @classmethod
    def ordered_values(cls) -> list[str]:
        return [cls.ANNUAL, cls.QUARTERLY, cls.MONTHLY, cls.WEEKLY, cls.DAILY]


def _date_time_query_to_values(date_time_query: DimensionQuery) -> tuple[str | None, str | None]:
    if date_time_query.operator == QueryOperator.BETWEEN:
        # Validate that we have at least 2 values for BETWEEN operator
        if len(date_time_query.values) < 2:
            _log.warning(f"BETWEEN operator requires 2 values, got {len(date_time_query.values)}")
            # Return first value as start if available
            return date_time_query.values[0] if date_time_query.values else None, None
        return date_time_query.values[0], date_time_query.values[1]
    elif date_time_query.operator == QueryOperator.GREATER_THAN_OR_EQUALS:
        return date_time_query.values[0] if date_time_query.values else None, None
    elif date_time_query.operator == QueryOperator.LESS_THAN_OR_EQUALS:
        return None, date_time_query.values[0] if date_time_query.values else None

    _log.warning(f"Unsupported date time operator: {date_time_query.operator}")
    return None, None


def _frequency_query_to_value(frequency_query: DimensionQuery) -> str | None:
    if frequency_query.operator not in [QueryOperator.IN, QueryOperator.ALL]:
        _log.warning(f"Unsupported frequency operator: {frequency_query.operator}")
        return None

    for freq in FrequencyEnum.ordered_values():
        if freq in frequency_query.values:
            return freq

    _log.warning(f"Unsupported frequency values: {frequency_query.values}")
    return None


def _adjust_start_date(value: str, frequency: FrequencyEnum) -> str:
    try:
        dt = date.fromisoformat(value)
    except ValueError:
        _log.warning(f"Invalid date format was passed to _adjust_start_date: {value}")
        return value
    if frequency == FrequencyEnum.ANNUAL:
        return dt.replace(month=1, day=1).isoformat()
    elif frequency == FrequencyEnum.QUARTERLY:
        quarter = (dt.month - 1) // 3 + 1
        month = (quarter - 1) * 3 + 1
        return dt.replace(month=month, day=1).isoformat()
    elif frequency == FrequencyEnum.MONTHLY:
        return dt.replace(day=1).isoformat()
    elif frequency == FrequencyEnum.WEEKLY:
        start_of_week = dt - timedelta(days=dt.weekday())
        return start_of_week.isoformat()
    elif frequency == FrequencyEnum.DAILY:
        return value
    else:
        _log.warning(f"Unsupported frequency: {frequency}")
        return value


def _adjust_end_date(value: str, frequency: FrequencyEnum) -> str:
    try:
        dt = date.fromisoformat(value)
    except ValueError:
        _log.warning(f"Invalid date format was passed to _adjust_end_date: {value}")
        return value
    if frequency == FrequencyEnum.ANNUAL:
        return dt.replace(month=12, day=31).isoformat()
    elif frequency == FrequencyEnum.QUARTERLY:
        quarter = (dt.month - 1) // 3 + 1
        month = quarter * 3
        if month == 12:
            # December - get last day directly
            last_day = 31
        else:
            last_day = (dt.replace(month=month + 1, day=1) - timedelta(days=1)).day
        return dt.replace(month=month, day=last_day).isoformat()
    elif frequency == FrequencyEnum.MONTHLY:
        if dt.month == 12:
            # December - get last day directly
            last_day = 31
        else:
            last_day = (dt.replace(month=dt.month + 1, day=1) - timedelta(days=1)).day
        return dt.replace(day=last_day).isoformat()
    elif frequency == FrequencyEnum.WEEKLY:
        end_of_week = dt + timedelta(days=(6 - dt.weekday()))
        return end_of_week.isoformat()
    elif frequency == FrequencyEnum.DAILY:
        return value
    else:
        _log.warning(f"Unsupported frequency: {frequency}")
        return value


def _expand_time_range_query(dataset: DataSet, query: DataSetQuery) -> DataSetQuery:
    date_time_dimension = dataset.get_time_dimension()
    frequency_dimension = dataset.get_frequency_dimension()

    date_time_query = next(
        (q for q in query.dimensions_queries if q.dimension_id == date_time_dimension.entity_id),
        None,
    )
    frequency_query = next(
        (q for q in query.dimensions_queries if q.dimension_id == frequency_dimension.entity_id),
        None,
    )

    if not date_time_query or not frequency_query:
        return query

    frequency = _frequency_query_to_value(frequency_query)
    if not frequency:
        return query

    start_date, end_date = _date_time_query_to_values(date_time_query)

    if start_date:
        start_date = _adjust_start_date(start_date, FrequencyEnum(frequency))
    if end_date:
        end_date = _adjust_end_date(end_date, FrequencyEnum(frequency))

    new_date_time_query = DimensionQuery(
        dimension_id=date_time_query.dimension_id,
        operator=date_time_query.operator,
        values=[v for v in [start_date, end_date] if v is not None],
        is_default=date_time_query.is_default,
    )
    return DataSetQuery(
        is_valid=query.is_valid,
        dimensions_queries=[
            *(
                q
                for q in query.dimensions_queries
                if q.dimension_id != date_time_dimension.entity_id
            ),
            new_date_time_query,
        ],
    )


def expand_time_range(inputs: dict) -> dict[str, DataSetQuery]:
    with debug_timer("_expand_time_range"):
        chain_state = ChainState(**inputs)

        result = {}

        for dataset_id, query in chain_state.dataset_queries.items():
            dataset = chain_state.datasets_dict[dataset_id]
            result[dataset_id] = _expand_time_range_query(dataset, query)

        return result

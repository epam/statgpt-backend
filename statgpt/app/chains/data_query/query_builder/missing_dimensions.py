"""The required dimensions a query does not specify yet, with the values available for each."""

from collections.abc import Iterator

from statgpt.app.schemas.data_query_outcome import (
    DimensionValueInfo,
    MissingDimensionInfo,
    MissingDimensionsInfo,
)
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import CategoricalDimension, DataSetAvailabilityQuery, DataSetQuery


def iter_missing_dimensions(
    dataset: VersionedDataSet,
    query: DataSetQuery,
    availability: DataSetAvailabilityQuery,
) -> Iterator[tuple[CategoricalDimension, list[DimensionValueInfo]]]:
    """Yield each required-but-unspecified categorical dimension with its available values.

    A dimension is "missing" when the query carries no filter for it. Only categorical
    dimensions with available values (given the rest of the query) are yielded, so the
    caller can offer the user concrete values to choose from.
    """
    missing_dimensions = [
        d for d in dataset.data.dimensions() if d.entity_id not in query.dimensions_queries_dict
    ]
    for dimension in missing_dimensions:
        if not isinstance(dimension, CategoricalDimension):
            continue
        available_values_query = availability.dimensions_queries_dict.get(dimension.entity_id)
        if available_values_query is None or not (values := available_values_query.values):
            logger.warning(
                f'There are no available values for dimension "{dimension.name}". '
                'Can\'t offer available values for user to select from.'
            )
            continue
        entities = {v.query_id: v for v in dimension.available_values}
        value_infos = [
            DimensionValueInfo(
                id=entities[value_id].query_id,
                name=entities[value_id].name,
                description=entities[value_id].description,
            )
            for value_id in values
        ]
        yield dimension, value_infos


def build_missing_dimensions_info(
    dataset_id: str,
    dataset: VersionedDataSet,
    query: DataSetQuery,
    availability: DataSetAvailabilityQuery,
) -> MissingDimensionsInfo:
    """Build the typed missing-dimensions payload for the tool's structured content."""
    dimensions = [
        MissingDimensionInfo(
            dimension_id=dimension.entity_id,
            name=dimension.name,
            available_values=value_infos,
        )
        for dimension, value_infos in iter_missing_dimensions(dataset, query, availability)
    ]
    return MissingDimensionsInfo(
        dataset_id=dataset_id, dataset_urn=dataset.data.source_id, dimensions=dimensions
    )

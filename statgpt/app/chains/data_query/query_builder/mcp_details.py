"""What the MCP structured content reports about the queries, collected by the chains that render
the response. It decorates the MCP response only, so a failure here must never break the answer
the user gets."""

import asyncio
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.data_query.query_builder import missing_dimensions
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.utils import dataset_utils
from statgpt.app.schemas.data_query_outcome import (
    DataQueryMcpPayload,
    InvalidPeriodInfo,
    MissingDimensionsInfo,
    QueryDetails,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.query_builder import ChainState
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataSetQuery, DateTimeDimension
from statgpt.common.data.base.enums import InvalidDataSetQueryReasonType
from statgpt.common.data.base.query import InvalidDataSetQueryReason
from statgpt.common.schemas.enums import InvocationSource

_REJECTED_BOUNDS: Mapping[str, Literal["startPeriod", "endPeriod"]] = {
    "selected_start": "startPeriod",
    "selected_end": "endPeriod",
}


class _InvalidTimePeriodDetails(BaseModel):
    """The `details` of an invalid time period reason, as the finalize chain records them."""

    field: Literal["selected_start", "selected_end"]
    value: str
    available_start: str | None = None
    available_end: str | None = None


def updated_mcp_payload(
    inputs: dict,
    *,
    message: str | None,
    query_details: dict[str, QueryDetails] | None = None,
    executed_at: str | None = None,
) -> DataQueryMcpPayload:
    """The MCP payload recorded so far, with the message and the given fields replaced."""
    payload = inputs.get(DataQueryParameters.MCP_PAYLOAD) or DataQueryMcpPayload()
    updates: dict[str, object] = {"message": message}
    if query_details is not None:
        updates["query_details"] = query_details
    if executed_at is not None:
        updates["executed_at"] = executed_at
    return payload.model_copy(update=updates)


async def query_details_for_mcp(
    inputs: dict, *, include_query: bool = False, valid_only: bool = False
) -> dict[str, QueryDetails]:
    """The query details, collected only for an MCP call: nothing else reads them."""
    if ChainParameters.get_invocation_source(inputs) is not InvocationSource.MCP:
        return {}
    return await _collect_query_details(
        ChainState.model_validate(inputs), include_query=include_query, valid_only=valid_only
    )


async def _collect_query_details(
    chain_state: ChainState, *, include_query: bool = False, valid_only: bool = False
) -> dict[str, QueryDetails]:
    """Details of the dataset queries in the state, by dataset id.

    `include_query` also records the constructed query, for outcomes whose queries did not run.
    `valid_only` skips the queries that cannot run. A query whose details fail is left out.
    """
    dataset_ids = [
        dataset_id
        for dataset_id, query in chain_state.dataset_queries.items()
        if query.is_valid or not valid_only
    ]
    details = await asyncio.gather(
        *(_query_details(chain_state, dataset_id, include_query) for dataset_id in dataset_ids)
    )
    return {
        dataset_id: query_details
        for dataset_id, query_details in zip(dataset_ids, details)
        if query_details is not None
    }


async def _query_details(
    chain_state: ChainState, dataset_id: str, include_query: bool
) -> QueryDetails | None:
    try:
        versioned_dataset = chain_state.datasets_dict[dataset_id]
        query = chain_state.dataset_queries[dataset_id]
        dataset = versioned_dataset.data
        citation = dataset.config.citation
        dimensions = dataset.dimensions()
        time_dimension_ids = {
            dimension.entity_id
            for dimension in dimensions
            if isinstance(dimension, DateTimeDimension)
        }
        default_ids = {q.dimension_id for q in query.dimensions_queries if q.is_default}
        json_query = dataset.to_json_query(query) if include_query else None

        return QueryDetails(
            dataset_urn=dataset.source_id,
            dataset_name=dataset.name,
            json_query=AppJsonQueryWithMetadata.from_common(json_query) if json_query else None,
            summary=query.short_summary,
            last_updated=await dataset_utils.dataset_last_updated(
                dataset, chain_state.auth_context
            ),
            provider=citation.provider if citation else None,
            is_official=dataset.config.is_official,
            dimension_names={dimension.entity_id: dimension.name for dimension in dimensions},
            default_dimension_ids=sorted(default_ids - time_dimension_ids),
            default_time_period=bool(default_ids & time_dimension_ids),
            invalid_period=_invalid_period(query.invalidity_reason),
            missing_dimensions=_missing_dimensions(chain_state, dataset_id, query),
        )
    except Exception:
        logger.exception(f"Failed to collect the MCP query details of dataset {dataset_id}")
        return None


def _invalid_period(reason: InvalidDataSetQueryReason | None) -> InvalidPeriodInfo | None:
    if reason is None or reason.type is not InvalidDataSetQueryReasonType.INVALID_TIME_PERIOD:
        return None
    details = _InvalidTimePeriodDetails.model_validate(reason.details)
    return InvalidPeriodInfo(
        rejected_bound=_REJECTED_BOUNDS[details.field],
        requested_value=details.value,
        available_start=details.available_start,
        available_end=details.available_end,
    )


def _missing_dimensions(
    chain_state: ChainState, dataset_id: str, query: DataSetQuery
) -> MissingDimensionsInfo | None:
    """The required dimensions the query does not specify, when that is why it is invalid: the
    query constructor marks such a query invalid without recording a reason."""
    if query.is_valid or query.invalidity_reason is not None:
        return None
    availability = chain_state.strong_availability.get(dataset_id)
    if availability is None:
        return MissingDimensionsInfo(
            dataset_id=dataset_id, dataset_urn=chain_state.datasets_dict[dataset_id].data.source_id
        )
    return missing_dimensions.build_missing_dimensions_info(
        dataset_id, chain_state.datasets_dict[dataset_id], query, availability
    )

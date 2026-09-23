"""What the MCP structured content reports about the queries, collected by the chains that render
the response. It decorates the MCP response only, so a failure here must never break the answer
the user gets."""

import asyncio
from typing import Any

from statgpt.app.chains.data_query.parameters import DataQueryParameters
from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.chains.utils import dataset_utils
from statgpt.app.schemas.data_query_outcome import (
    DataQueryMcpPayload,
    InvalidPeriodInfo,
    QueryDetails,
)
from statgpt.app.schemas.query import AppJsonQueryWithMetadata
from statgpt.app.schemas.query_builder import ChainState
from statgpt.app.services.chat_facade import VersionedDataSet
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.config import multiline_logger as logger
from statgpt.common.data.base import DataSetQuery, DateTimeDimension
from statgpt.common.data.base.enums import InvalidDataSetQueryReasonType
from statgpt.common.data.base.query import InvalidDataSetQueryReason
from statgpt.common.schemas.enums import InvocationSource

_REJECTED_BOUNDS = {"selected_start": "start", "selected_end": "end"}


def updated_mcp_payload(inputs: dict, **updates: Any) -> DataQueryMcpPayload:
    """The MCP payload recorded so far, with `updates` applied."""
    payload = inputs.get(DataQueryParameters.MCP_PAYLOAD) or DataQueryMcpPayload()
    return payload.model_copy(update=updates)


async def query_details_for_mcp(
    inputs: dict, *, include_query: bool = False
) -> dict[str, QueryDetails]:
    """The query details, collected only for an MCP call: nothing else reads them."""
    if ChainParameters.get_invocation_source(inputs) is not InvocationSource.MCP:
        return {}
    return await collect_query_details(
        ChainState.model_validate(inputs), include_query=include_query
    )


async def collect_query_details(
    chain_state: ChainState, *, include_query: bool = False
) -> dict[str, QueryDetails]:
    """Details of every dataset query in the state, by dataset id; empty on failure.

    `include_query` also records the constructed query, for outcomes whose queries did not run.
    """
    try:
        dataset_ids = list(chain_state.dataset_queries)
        details = await asyncio.gather(
            *(
                _query_details(
                    chain_state.datasets_dict[dataset_id],
                    chain_state.dataset_queries[dataset_id],
                    chain_state.auth_context,
                    include_query,
                )
                for dataset_id in dataset_ids
            )
        )
        return dict(zip(dataset_ids, details))
    except Exception:
        logger.exception("Failed to collect the query details for the MCP response")
        return {}


async def _query_details(
    versioned_dataset: VersionedDataSet,
    query: DataSetQuery,
    auth_context: AuthContext,
    include_query: bool,
) -> QueryDetails:
    dataset = versioned_dataset.data
    citation = dataset.config.citation
    time_dimension_ids = {
        dimension.entity_id
        for dimension in dataset.dimensions()
        if isinstance(dimension, DateTimeDimension)
    }
    default_ids = {q.dimension_id for q in query.dimensions_queries if q.is_default}
    json_query = dataset.to_json_query(query) if include_query else None

    return QueryDetails(
        dataset_urn=dataset.source_id,
        dataset_name=dataset.name,
        json_query=AppJsonQueryWithMetadata.from_common(json_query) if json_query else None,
        summary=query.short_summary,
        last_updated=await dataset_utils.dataset_last_updated(dataset, auth_context),
        provider=citation.provider if citation else None,
        is_official=dataset.config.is_official,
        dimension_names={dimension.entity_id: dimension.name for dimension in dataset.dimensions()},
        default_dimension_ids=sorted(default_ids - time_dimension_ids),
        default_time_period=bool(default_ids & time_dimension_ids),
        invalid_period=_invalid_period(query.invalidity_reason),
    )


def _invalid_period(reason: InvalidDataSetQueryReason | None) -> InvalidPeriodInfo | None:
    if reason is None or reason.type is not InvalidDataSetQueryReasonType.INVALID_TIME_PERIOD:
        return None
    details = reason.details
    available_start, available_end = details.get("available_start"), details.get("available_end")
    return InvalidPeriodInfo(
        rejected_bound=_REJECTED_BOUNDS[details["field"]],  # type: ignore[arg-type]
        requested_value=str(details["value"]),
        available_start=str(available_start) if available_start is not None else None,
        available_end=str(available_end) if available_end is not None else None,
    )

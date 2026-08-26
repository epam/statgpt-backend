"""Referring to discovery datasets when the data query pipeline finds no data.

The entry condition and the orchestration live here rather than in the tool, so the tool keeps
one call and this stays testable without a query-builder run.
"""

import logging

from statgpt.app.chains.discovery.referral import render_referral
from statgpt.app.chains.discovery.search import DiscoverySearchService
from statgpt.app.schemas.data_query_outcome import DataQueryStatus
from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import DiscoveryFallbackConfig

_log = logging.getLogger(__name__)


def is_data_query_miss(status: DataQueryStatus, config: DiscoveryFallbackConfig) -> bool:
    """Whether this outcome means StatGPT looked for data and found none.

    Only two of the pipeline's outcomes do. `dataset_selection_required` and
    `missing_dimensions` mean Grade A can still answer once the user supplies something, and
    `invalid_time_period` means the request needs correcting - referring the user elsewhere in
    any of those cases would abandon an answer that is still available.

    `failed` is excluded on purpose. It means a fetch errored, so StatGPT does not know whether
    the data exists; telling the user it lives elsewhere would be a guess dressed as help.
    """
    if status is DataQueryStatus.NO_DATA:
        return True
    if status is DataQueryStatus.EXECUTED_NO_DATA:
        return config.on_executed_no_data
    return False


async def refer_to_discovery(
    *,
    question: str,
    status: DataQueryStatus,
    countries: list[str],
    config: DiscoveryFallbackConfig,
    data_service: ChannelServiceFacade,
    auth_context: AuthContext,
) -> str:
    """Search the discovery index and render a referral, or return an empty string.

    Empty covers every reason there is nothing to add: the fallback is off, the outcome was not a
    miss, the channel has no discovery application configured, nothing was retrieved, or the judge
    found nothing relevant. The caller appends the result unconditionally, so all of those look
    the same from the tool's side - a no-data answer, unchanged.

    Nothing here raises. A referral is an extra on top of an answer the user is already getting,
    so a discovery failure is logged and dropped rather than turned into a failed tool call.
    """
    if not config.enabled or not is_data_query_miss(status, config):
        return ""

    discovery_rag = data_service.channel_config.discovery_rag
    if discovery_rag is None:
        _log.info(
            "Discovery fallback is enabled but the channel configures no discovery RAG"
            " application (`details.discoveryRag.applicationId`); skipping the referral"
        )
        return ""

    try:
        async with DiscoverySearchService(
            config=config,
            application_id=discovery_rag.get_application_id(),
            statgpt_channel=data_service.deployment_id,
            auth_context=auth_context,
        ) as service:
            result = await service.search(question, countries)
    except Exception:
        _log.exception("Discovery fallback failed; answering without a referral")
        return ""

    _log.info(
        f"Discovery fallback: retrieved {result.retrieved} dataset(s)"
        f" within areas {result.grounded_areas or 'unfiltered'},"
        f" referring to {len(result.items)}"
    )
    return render_referral(result)

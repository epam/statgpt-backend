from statgpt.app.schemas import ChannelDatasetsMetadataResponse
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.models.database import get_readonly_session_context_manager
from statgpt.common.schemas import Channel
from statgpt.common.services.dataset import DataSetService


async def build_channel_datasets_metadata(
    channel: Channel, auth_context: AuthContext
) -> ChannelDatasetsMetadataResponse:
    """Build the channel datasets metadata payload shared by the `/metadata/datasets` service
    endpoint and the MCP-App-only datasets-metadata tool, so both stay in sync."""
    async with get_readonly_session_context_manager() as session:
        datasets = await DataSetService(session).get_channel_dataset_schemas_with_last_updated(
            limit=None,
            offset=0,
            channel_id=channel.id,
            auth_context=auth_context,
        )

    for ds in datasets:
        resolved = ds.last_completed_version and ds.last_completed_version.resolved_config
        if resolved:
            ds.dataset = ds.dataset.model_copy(update={"details": resolved})

    return ChannelDatasetsMetadataResponse(
        deployment_id=channel.deployment_id,
        title=channel.title,
        n_datasets=len(datasets),
        datasets=datasets,
    )

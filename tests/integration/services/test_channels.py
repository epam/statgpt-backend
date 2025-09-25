import pytest
from fastapi import HTTPException
from sqlalchemy.exc import NoResultFound

from admin_portal.auth.auth_context import SystemUserAuthContext
from admin_portal.services import AdminPortalChannelService as ChannelService
from common import schemas
from common.schemas.data_query_tool import DataQueryPrompts
from common.settings.langchain import langchain_settings

# ~~~~~ Channel Tests ~~~~~


@pytest.mark.asyncio
async def test_get_channel_by_id_not_found(session, clear_channels):
    channel_service = ChannelService(session)

    with pytest.raises(HTTPException) as e:
        await channel_service.get_schema_by_id(123456789)

    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_get_channel_by_deployment_id_not_found(session, clear_channels):
    channel_service = ChannelService(session)

    with pytest.raises(NoResultFound):
        await channel_service.get_channel_by_deployment_id('unknown_id')


@pytest.mark.asyncio
async def test_add_channel(session, clear_channels):
    channel_service = ChannelService(session)

    channel = schemas.ChannelBase(
        title='test_title',
        description='test_description',
        deployment_id='test_deployment_id',
        details=schemas.ChannelConfig(
            supreme_agent=schemas.SupremeAgentConfig(
                name="Test",
                domain="Test domain",
                terminology_domain="Test terminology_domain",
                language_instructions=["Test instruction"],
            ),
            data_query=schemas.DataQueryTool(
                name="Test_Query_Builder",
                description="Test Query Builder Description",
                details=schemas.DataQueryDetails(
                    prompts=DataQueryPrompts(
                        datetime_prompt='Testing datetime_prompt',
                        indicators_selection_system_prompt='indicator_prompt_test',
                    ),
                ),
            ),
            named_entity_types=["Test entity"],
        ),
        llm_model=langchain_settings.embedding_default_model.value,
    )

    res = await channel_service.create_channel(channel)

    assert res.title == "test_title"
    assert res.description == "test_description"
    assert res.deployment_id == "test_deployment_id"
    assert res.llm_model == langchain_settings.embedding_default_model.value
    assert res.details.named_entity_types == ["Test entity"]

    assert res.details.available_datasets is None
    assert res.details.file_rag is None

    assert res.details.data_query.name == "Test_Query_Builder"
    assert res.details.data_query.description == "Test Query Builder Description"

    details = res.details.data_query.details
    assert details.version == schemas.DataQueryVersion.v2
    assert details.indexer_version == schemas.IndexerVersion.semantic
    assert details.indicator_selection_version == schemas.IndicatorSelectionVersion.semantic_v4
    assert details.prompts.datetime_prompt == "Testing datetime_prompt"
    assert details.prompts.group_expander_prompt is None
    assert details.prompts.group_expander_fallback_prompt is None
    assert details.prompts.indicators_selection_system_prompt == "indicator_prompt_test"

    res2 = await channel_service.get_schema_by_id(res.id)
    assert res2 == res


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'deployment_id, indexer_version',
    (
        ['test_deployment_id_5', schemas.IndexerVersion.semantic],
        ['test_deployment_id_6', schemas.IndexerVersion.hybrid],
    ),
)
async def test_add_channel_with_v2_versions(
    session, clear_channels, deployment_id, indexer_version
):
    channel_service = ChannelService(session)

    channel = schemas.ChannelBase(
        title='test_title',
        description='test_description',
        deployment_id=deployment_id,
        details=schemas.ChannelConfig(
            supreme_agent=schemas.SupremeAgentConfig(
                name="Test",
                domain="Test domain",
                terminology_domain="Test terminology_domain",
                language_instructions=["Test instruction"],
            ),
            data_query=schemas.DataQueryTool(
                name="Test_Query_Builder",
                description="Test Query Builder Description",
                details=schemas.DataQueryDetails(
                    version=schemas.DataQueryVersion.v2,
                    indexer_version=indexer_version,
                ),
            ),
        ),
        llm_model=langchain_settings.embedding_default_model.value,
    )

    res = await channel_service.create_channel(channel)
    assert res.deployment_id == deployment_id
    assert res.details.data_query.details.version == schemas.DataQueryVersion.v2
    assert res.details.data_query.details.indexer_version == indexer_version

    res2 = await channel_service.get_schema_by_id(res.id)
    assert res2 == res


@pytest.mark.asyncio
async def test_channel_service_get_list(session, clear_channels):
    channel_service = ChannelService(session)

    count = await channel_service.get_channels_count()
    assert count == 0

    channels = await channel_service.get_channels_schemas(100, 0)
    assert channels == []

    channel1 = await channel_service.create_channel(
        schemas.ChannelBase(
            title='test_title_1',
            deployment_id='test_deployment_id_1',
            llm_model=langchain_settings.embedding_default_model.value,
            details=schemas.ChannelConfig(
                supreme_agent=schemas.SupremeAgentConfig(
                    name="Test",
                    domain="Test domain",
                    terminology_domain="Test terminology_domain",
                    language_instructions=["Test instruction"],
                ),
            ),
        )
    )

    channel2 = await channel_service.create_channel(
        schemas.ChannelBase(
            title='test_title_2',
            deployment_id='test_deployment_id_2',
            llm_model=langchain_settings.embedding_default_model.value,
            details=schemas.ChannelConfig(
                supreme_agent=schemas.SupremeAgentConfig(
                    name="Test",
                    domain="Test domain",
                    terminology_domain="Test terminology_domain",
                    language_instructions=["Test instruction"],
                ),
            ),
        )
    )

    count = await channel_service.get_channels_count()
    assert count == 2

    channels = await channel_service.get_channels_schemas(100, 0)
    assert channels == [channel1, channel2]


@pytest.mark.asyncio
async def test_update_channel(session, clear_channels):
    channel_service = ChannelService(session)

    channel = schemas.ChannelBase(
        title='test_title_2',
        description='Test description 2',
        deployment_id='test_deployment_id_2',
        details=schemas.ChannelConfig(
            supreme_agent=schemas.SupremeAgentConfig(
                name="Test",
                domain="Test domain",
                terminology_domain="Test terminology_domain",
                language_instructions=["Test instruction"],
            ),
        ),
        llm_model=langchain_settings.embedding_default_model.value,
    )

    res = await channel_service.create_channel(channel)

    assert res.details.data_query is None
    assert len(res.details.named_entity_types) == 0

    res2 = await channel_service.update(
        res.id,
        schemas.ChannelUpdate(
            title='new_title',
            description='New description.',
            llm_model=langchain_settings.embedding_default_model.value,
            details=schemas.ChannelConfig(
                supreme_agent=schemas.SupremeAgentConfig(
                    name="Test",
                    domain="Test domain",
                    terminology_domain="Test terminology_domain",
                    language_instructions=["Test instruction"],
                ),
                data_query=schemas.DataQueryTool(
                    name="Test_Query_Builder_Update",
                    description="Test Query Builder Update Description",
                    details=schemas.DataQueryDetails(
                        indexer_version=schemas.IndexerVersion.hybrid,
                        prompts=DataQueryPrompts(
                            group_expander_prompt="Test group_expander_prompt",
                        ),
                    ),
                ),
                named_entity_types=["Country", "Unit of measure"],
            ),
        ),
    )

    assert res2.title == "new_title"
    assert res2.description == "New description."
    assert res2.deployment_id == "test_deployment_id_2"
    assert res2.llm_model == langchain_settings.embedding_default_model.value

    assert res2.details.data_query.details.version == schemas.DataQueryVersion.v2
    assert res2.details.data_query.details.indexer_version == schemas.IndexerVersion.hybrid
    assert res2.details.data_query.details.prompts.datetime_prompt is None
    assert (
        res2.details.data_query.details.prompts.group_expander_prompt
        == "Test group_expander_prompt"
    )
    assert res2.details.data_query.details.prompts.group_expander_fallback_prompt is None
    assert res2.details.data_query.details.prompts.indicators_selection_system_prompt is None
    assert sorted(res2.details.named_entity_types) == ["Country", "Unit of measure"]

    # it is very important to check that the object was updated in the database:
    res3 = await channel_service.get_schema_by_id(res.id)
    assert res3 == res2


# @pytest.mark.asyncio
@pytest.mark.skip(reason="Failed after removing local embeddings model")
async def test_delete_channel(session, clear_channels):
    channel_service = ChannelService(session)

    res = await channel_service.create_channel(
        schemas.ChannelBase(
            title='test_title_3',
            deployment_id='test_deployment_id_3',
            llm_model=langchain_settings.embedding_default_model.value,
            details=schemas.ChannelConfig(
                supreme_agent=schemas.SupremeAgentConfig(
                    name="Test",
                    domain="Test domain",
                    terminology_domain="Test terminology_domain",
                    language_instructions=["Test instruction"],
                ),
            ),
        )
    )

    channels = await channel_service.get_channels_schemas(100, 0)
    assert channels == [res]

    await channel_service.delete(res.id, auth_context=SystemUserAuthContext())

    with pytest.raises(HTTPException) as e:
        await channel_service.get_schema_by_id(res.id)

    assert e.value.status_code == 404

    channels = await channel_service.get_channels_schemas(100, 0)
    assert channels == []

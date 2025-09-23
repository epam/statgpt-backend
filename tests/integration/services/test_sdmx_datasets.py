import uuid

import pytest
import pytest_asyncio  # noqa: F401

from admin_portal.auth.auth_context import SystemUserAuthContext
from admin_portal.services import AdminPortalChannelService as ChannelService
from admin_portal.services import AdminPortalDataSetService as DataSetService
from admin_portal.services import AdminPortalDataSourceService as DataSourceService
from admin_portal.services.dataset import reload_indicators_in_background_task
from common import schemas
from common.config import LangChainConfig
from common.data.base import DatasetCitation, IndexerConfig
from common.data.base.dataset import IndexerIndicatorConfig

from .mocks import BackgroundTasksMock

# ~~~~~ Tools ~~~~~


async def get_data_source(session) -> schemas.DataSource:
    data_source_service = DataSourceService(session)
    data_source = await data_source_service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source',
            type_id=2,
            details={
                'apiKey': 'test_key',
                'sdmxConfig': {
                    'id': 'test_id',
                    'name': 'test_name',
                    'url': 'https://example.com/sdmx.url',
                },
            },
        )
    )
    return data_source


async def get_channel(
    session, indexer_version: schemas.IndexerVersion = schemas.IndexerVersion.semantic
) -> schemas.Channel:
    channel_service = ChannelService(session)
    channel = await channel_service.create_channel(
        schemas.ChannelBase(
            title='test_title_1',
            deployment_id='test_deployment_id_1',
            llm_model=LangChainConfig.DEFAULT_EMBEDDINGS_MODEL.value,
            details=schemas.ChannelConfig(
                supreme_agent=schemas.SupremeAgentConfig(
                    name="Test",
                    domain="Test domain",
                    terminology_domain="Test terminology domain",
                    language_instructions=["Test instruction"],
                ),
                data_query=schemas.DataQueryTool(
                    name="Test_Query_Builder",
                    description="Test Query Builder Description",
                    details=schemas.DataQueryDetails(
                        indexer_version=indexer_version,
                    ),
                ),
            ),
        )
    )
    return channel


# ~~~~~ Dataset Tests ~~~~~


@pytest.mark.asyncio
async def test_create_dataset(session, clear_all, sdmx_clint_mock):
    data_source = await get_data_source(session)
    random_uuid = uuid.uuid4()

    dataset_service = DataSetService(session)
    dataset = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid,
            title='CPI',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(4.0.0)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
                'useTitleFromSrc': True,
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    assert dataset.id_ == random_uuid
    assert dataset.data_source_id == data_source.id
    assert dataset.details['urn'] == 'IMF.STA:CPI(4.0.0)'
    assert dataset.details['indicatorDimensions'] == [
        'INDEX_TYPE',
        'COICOP_1999',
        'TYPE_OF_TRANSFORMATION',
    ]
    assert dataset.details['citation'] is None
    assert dataset.details['indexer'] is None

    assert dataset.title == "Consumer Price Index (CPI)"
    assert dataset.description.startswith(
        "The Consumer Price Index (CPI) dataset includes the national consumer price indexes by economy."
    )
    assert dataset.description.endswith(
        "which households use directly, or indirectly, to satisfy their own needs and wants."
    )

    res = await dataset_service.get_schema_by_id(dataset.id, auth_context=SystemUserAuthContext())

    assert res.data_source.id == data_source.id
    assert res.data_source.title == data_source.title
    assert res.data_source.type_id == data_source.type_id

    res.data_source = None
    assert res == dataset


@pytest.mark.asyncio
async def test_update_dataset(session, clear_all, sdmx_clint_mock):
    data_source = await get_data_source(session)
    random_uuid = uuid.uuid4()

    dataset_service = DataSetService(session)
    dataset = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid,
            title='CPI',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(4.0.0)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
                'useTitleFromSrc': False,
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    assert dataset.id_ == random_uuid
    assert dataset.title == 'CPI'
    assert dataset.data_source_id == data_source.id
    assert dataset.details['urn'] == 'IMF.STA:CPI(4.0.0)'
    assert dataset.details['citation'] is None

    citation = DatasetCitation(
        provider="International Monetary Fund",
        last_updated="July 2025",
        url="https://data.imf.org/en/Data-Explorer?datasetUrn=IMF.STA:CPI(4.0.0)",
    )

    indexer_config = IndexerConfig(
        description='Test description',
        indicator=IndexerIndicatorConfig(unpack=True),
    )

    dataset = await dataset_service.update(
        dataset.id,
        schemas.DataSetUpdate(
            title='CPI Updated',
            details={
                'urn': 'IMF.STA:CPI(3.0.1)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
                'citation': citation.model_dump(),
                'indexer': indexer_config.model_dump(),
                'useTitleFromSrc': False,
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    assert dataset.id_ == random_uuid
    assert dataset.title == 'CPI Updated'
    assert dataset.details['urn'] == 'IMF.STA:CPI(3.0.1)'
    assert dataset.details['citation'] == citation.model_dump()
    assert dataset.details['indexer'] == indexer_config.model_dump(by_alias=True)

    res = await dataset_service.get_schema_by_id(dataset.id, auth_context=SystemUserAuthContext())
    assert res.id_ == random_uuid
    assert res.title == 'CPI Updated'
    assert res.details == dataset.details


@pytest.mark.asyncio
async def test_get_list_and_count(session, clear_all, sdmx_clint_mock):
    data_source = await get_data_source(session)
    random_uuid1 = uuid.uuid4()
    random_uuid2 = uuid.uuid4()

    dataset_service = DataSetService(session)

    count1 = await dataset_service.get_datasets_count(None, None)
    assert count1 == 0

    datasets1 = await dataset_service.get_datasets_schemas(
        limit=100, offset=0, auth_context=SystemUserAuthContext()
    )
    assert datasets1 == []

    ds1 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid1,
            title='CPI v4.0.0',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(4.0.0)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    count2 = await dataset_service.get_datasets_count(None, None)
    assert count2 == 1

    datasets2 = await dataset_service.get_datasets_schemas(
        limit=100, offset=0, auth_context=SystemUserAuthContext()
    )
    datasets2[0].data_source = None
    assert datasets2[0] == ds1

    ds2 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid2,
            title='CPI v3.0.1',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(3.0.1)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    count3 = await dataset_service.get_datasets_count(None, None)
    assert count3 == 2

    datasets3 = await dataset_service.get_datasets_schemas(
        limit=100, offset=0, auth_context=SystemUserAuthContext()
    )
    datasets3[0].data_source, datasets3[1].data_source = None, None
    assert datasets3 == [ds1, ds2]


# ~~~~~ Channel Dataset Tests ~~~~~


@pytest.mark.asyncio
async def test_create_channel_dataset(session, clear_all, sdmx_clint_mock):
    channel = await get_channel(session)
    data_source = await get_data_source(session)
    random_uuid = uuid.uuid4()

    dataset_service = DataSetService(session)
    dataset = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid,
            title='CPI',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(4.0.0)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    channel_dataset = await dataset_service.add_dataset_to_channel(channel.id, dataset.id)

    assert channel_dataset.channel_id == channel.id
    assert channel_dataset.dataset_id == dataset.id
    assert channel_dataset.preprocessing_status == schemas.PreprocessingStatusEnum.NOT_STARTED

    res = await dataset_service.get_channel_dataset_schema(channel.id, dataset.id)
    assert channel_dataset == res


@pytest.mark.asyncio
async def test_get_channel_datasets_and_count(session, clear_all, sdmx_clint_mock):
    channel = await get_channel(session)
    data_source = await get_data_source(session)
    random_uuid1 = uuid.uuid4()
    random_uuid2 = uuid.uuid4()

    dataset_service = DataSetService(session)

    count1 = await dataset_service.get_channel_datasets_count(channel.id)
    assert count1 == 0

    channel_datasets1 = await dataset_service.get_channel_dataset_schemas(
        limit=100, offset=0, channel_id=channel.id, auth_context=SystemUserAuthContext()
    )
    assert channel_datasets1 == []

    ds1 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid1,
            title='CPI v4.0.0',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(4.0.0)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    channel_dataset1 = await dataset_service.add_dataset_to_channel(channel.id, ds1.id)

    count2 = await dataset_service.get_channel_datasets_count(channel.id)
    assert count2 == 1

    channel_datasets2 = await dataset_service.get_channel_dataset_schemas(
        limit=100, offset=0, channel_id=channel.id, auth_context=SystemUserAuthContext()
    )
    assert len(channel_datasets2) == 1
    assert channel_datasets2[0].id == channel_dataset1.id
    assert channel_datasets2[0].dataset_id == ds1.id
    assert channel_datasets2[0].channel_id == channel.id

    ds2 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid2,
            title='CPI v3.0.1',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(3.0.1)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    channel_dataset2 = await dataset_service.add_dataset_to_channel(channel.id, ds2.id)

    count3 = await dataset_service.get_channel_datasets_count(channel.id)
    assert count3 == 2

    channel_datasets3 = await dataset_service.get_channel_dataset_schemas(
        limit=100, offset=0, channel_id=channel.id, auth_context=SystemUserAuthContext()
    )
    assert len(channel_datasets3) == 2
    assert channel_datasets3[0].id == channel_dataset1.id
    assert channel_datasets3[0].dataset_id == ds1.id
    assert channel_datasets3[0].channel_id == channel.id
    assert channel_datasets3[1].id == channel_dataset2.id
    assert channel_datasets3[1].dataset_id == ds2.id
    assert channel_datasets3[1].channel_id == channel.id

    # ~~~ Test remove channel dataset ~~~
    # TODO: fix code below

    # await dataset_service.remove_channel_dataset(channel.id, ds1.id)
    #
    # with pytest.raises(HTTPException) as e:
    #     await dataset_service.get_channel_dataset_schema(channel.id, ds1.id)
    #
    # assert e.value.status_code == 404
    #
    # count4 = await dataset_service.get_channel_datasets_count(channel.id)
    # assert count4 == 1
    #
    # channel_datasets4 = await dataset_service.get_channel_dataset_schemas(
    #     limit=100, offset=0, channel_id=channel.id, auth_context=SystemUserAuthContext()
    # )
    #
    # assert len(channel_datasets4) == 1
    # assert channel_datasets4[0].id == channel_dataset2.id
    # assert channel_datasets4[0].dataset_id == ds2.id
    # assert channel_datasets4[0].channel_id == channel.id


@pytest.mark.asyncio
async def test_reload_indicators(session, clear_all, sdmx_clint_mock):
    channel = await get_channel(session)
    data_source = await get_data_source(session)
    random_uuid = uuid.uuid4()

    dataset_service = DataSetService(session)

    dataset = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid,
            title='CPI',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(4.0.0)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    await dataset_service.add_dataset_to_channel(channel.id, dataset.id)

    background_tasks = BackgroundTasksMock()
    res = await dataset_service.reload_indicators(
        background_tasks,  # type: ignore
        channel.id,
        dataset.id,
        reindex_dimensions=True,
        harmonize_indicator=False,
        reindex_indicators=False,
        auth_context=SystemUserAuthContext(),
        max_n_embeddings=5,
    )

    assert res.preprocessing_status == schemas.PreprocessingStatusEnum.QUEUED

    func, args, kwargs = background_tasks.tasks[0]
    assert func == reload_indicators_in_background_task
    assert kwargs['channel_id'] == channel.id
    assert kwargs['dataset_id'] == dataset.id
    assert kwargs['previous_status'] == schemas.PreprocessingStatusEnum.NOT_STARTED
    assert kwargs['reindex_dimensions'] is True
    assert kwargs['reindex_indicators'] is False
    assert kwargs['max_n_embeddings'] == 5

    res2 = await dataset_service.get_channel_dataset_schema(channel.id, dataset.id)
    assert res2.preprocessing_status == schemas.PreprocessingStatusEnum.QUEUED

    # ~~~ Testing the background task ~~~

    # TODO: implement this after the proper mock is created


@pytest.mark.asyncio
async def test_reload_all_indicators(session, clear_all, sdmx_clint_mock):
    channel = await get_channel(session)
    data_source = await get_data_source(session)
    random_uuid1 = uuid.uuid4()
    random_uuid2 = uuid.uuid4()

    dataset_service = DataSetService(session)

    ds1 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid1,
            title='CPI v4.0.0',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(4.0.0)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    ds2 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid2,
            title='CPI v3.0.1',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(3.0.1)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    await dataset_service.add_dataset_to_channel(channel.id, ds1.id)
    await dataset_service.add_dataset_to_channel(channel.id, ds2.id)

    background_tasks = BackgroundTasksMock()
    res = await dataset_service.reload_all_indicators(
        background_tasks,  # type: ignore
        channel.id,
        reindex_dimensions=True,
        harmonize_indicator=False,
        reindex_indicators=True,
        auth_context=SystemUserAuthContext(),
        max_n_embeddings=5,
    )

    assert len(res) == 2
    assert res[0].preprocessing_status == schemas.PreprocessingStatusEnum.QUEUED
    assert res[1].preprocessing_status == schemas.PreprocessingStatusEnum.QUEUED

    assert len(background_tasks.tasks) == 2
    for f, args, kwargs in background_tasks.tasks:
        assert f == reload_indicators_in_background_task
        assert kwargs['channel_id'] == channel.id
        assert kwargs['dataset_id'] in [ds1.id, ds2.id]
        assert kwargs['previous_status'] == schemas.PreprocessingStatusEnum.NOT_STARTED
        assert kwargs['reindex_dimensions'] is True
        assert kwargs['reindex_indicators'] is True
        assert kwargs['max_n_embeddings'] == 5

    res2 = await dataset_service.get_channel_dataset_schemas(
        limit=100, offset=0, channel_id=channel.id, auth_context=SystemUserAuthContext()
    )
    assert len(res2) == 2
    assert res2[0].preprocessing_status == schemas.PreprocessingStatusEnum.QUEUED
    assert res2[1].preprocessing_status == schemas.PreprocessingStatusEnum.QUEUED


# @pytest.mark.asyncio
@pytest.mark.skip(reason="failed after fixes of old indexer")
async def test_reload_channel_dataset_in_background_v1(session, clear_all, sdmx_clint_mock):
    channel = await get_channel(session, indexer_version=schemas.IndexerVersion.semantic)
    data_source = await get_data_source(session)
    random_uuid = uuid.uuid4()

    dataset_service = DataSetService(session)

    dataset = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=random_uuid,
            title='CPI',
            data_source_id=data_source.id,
            details={
                'urn': 'IMF.STA:CPI(4.0.0)',
                'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    channel_dataset = await dataset_service.add_dataset_to_channel(channel.id, dataset.id)

    assert channel_dataset.channel_id == channel.id
    assert channel_dataset.dataset_id == dataset.id
    assert channel_dataset.preprocessing_status == schemas.PreprocessingStatusEnum.NOT_STARTED

    await dataset_service.reload_channel_dataset_in_background(
        channel.id,
        dataset.id,
        reindex_dimensions=True,
        harmonize_indicator=False,
        reindex_indicators=True,
        auth_context=SystemUserAuthContext(),
        previous_status=schemas.PreprocessingStatusEnum.COMPLETED,
        # (COMPLETED status is needed to clear previous data)
        max_n_embeddings=5,
    )

    updated_channel_dataset = await dataset_service.get_channel_dataset_schema(
        channel.id, dataset.id
    )

    assert updated_channel_dataset.preprocessing_status == schemas.PreprocessingStatusEnum.COMPLETED


# @pytest.mark.asyncio
# async def test_reload_channel_dataset_in_background_v2(session, clear_all, sdmx_clint_mock):
#
#     channel = await get_channel(session, indexer_version=schemas.IndexerVersion.v2)
#     data_source = await get_data_source(session)
#
#     dataset_service = DataSetService(session)
#
#     indexer_config = IndexerConfig(
#         topics=['test_topic_1', 'test_topic_2'],
#         description='Test description',
#         indicator=IndexerIndicatorConfig(unpack=True),
#     )
#
#     dataset = await dataset_service.create_dataset(
#         schemas.DataSetBase(
#             data_source_id=data_source.id,
#             details={
#                 'urn': 'IMF.STA:CPI(4.0.0)',
#                 'indicatorDimensions': ['INDEX_TYPE', 'COICOP_1999', 'TYPE_OF_TRANSFORMATION'],
#                 'indexer': indexer_config.dict(),
#             },
#         ),
#         auth_context=SystemUserAuthContext(),
#     )
#
#     assert dataset.details['indexer'] == indexer_config.dict(by_alias=True)
#
#     channel_dataset = await dataset_service.add_dataset_to_channel(channel.id, dataset.id)
#
#     assert channel_dataset.channel_id == channel.id
#     assert channel_dataset.dataset_id == dataset.id
#     assert channel_dataset.preprocessing_status == schemas.PreprocessingStatusEnum.NOT_STARTED
#
#     await dataset_service.reload_channel_dataset_in_background(
#         channel.id,
#         dataset.id,
#         reindex_dimensions=True,
#         reindex_indicators=True,
#         previous_status=schemas.PreprocessingStatusEnum.COMPLETED,
#         # (COMPLETED status is needed to clear previous data)
#         max_n_embeddings=5,
#     )
#
#     updated_channel_dataset = await dataset_service.get_channel_dataset_schema(
#         channel.id, dataset.id
#     )
#
#     assert updated_channel_dataset.preprocessing_status == schemas.PreprocessingStatusEnum.COMPLETED

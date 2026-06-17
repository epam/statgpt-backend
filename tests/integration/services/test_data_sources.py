import uuid

import pytest
import pytest_asyncio  # noqa: F401
from fastapi import HTTPException
from sqlalchemy import select

from statgpt.admin.auth.auth_context import SystemUserAuthContext
from statgpt.admin.services import AdminPortalDataSetService as DataSetService
from statgpt.admin.services import AdminPortalDataSourceDeletionService as DataSourceDeletionService
from statgpt.admin.services import AdminPortalDataSourceService as DataSourceService
from statgpt.admin.services.exceptions import DatasetInUseError
from statgpt.common import schemas
from statgpt.common.data.sdmx import Sdmx21DataSourceHandler, SdmxDataSourceConfig
from statgpt.common.data.sdmx.common.config import SdmxHeaders, SdmxSupport
from statgpt.common.models import DataSourceType
from statgpt.common.services import DataSourceTypeService

from .conftest import get_audit_logs
from .test_sdmx_datasets import get_channel

# ~~~~~ Data Source Type Tests ~~~~~


@pytest.mark.asyncio
async def test_data_source_types(session):
    """The database created by alembic migration always has three data source types."""

    service = DataSourceTypeService(session)

    count = await service.get_count()
    assert count == 3

    types = await service.get_data_source_types(100, 0)
    assert {ds_type.name for ds_type in types} == {"SDMX21", "QH_SDMX21", "PROXY_SDMX30"}


@pytest.mark.asyncio
async def test_sdmx_data_source_type(session):
    """The database created by alembic migration always has one type of data source."""

    service = DataSourceTypeService(session)

    ds_type = await service.get_by_id(1)
    assert ds_type.id == 1
    assert ds_type.name == 'SDMX21'

    config_class = await service.get_config_class(1)
    assert config_class == SdmxDataSourceConfig

    handler_class = await service.get_data_source_handler_class_by_id(1)
    assert handler_class == Sdmx21DataSourceHandler


# ~~~~~ Data Source Tests ~~~~~


async def get_qh_sdmx21_ds_type(session):
    result = await session.execute(select(DataSourceType).where(DataSourceType.name == "QH_SDMX21"))
    data_source_type = result.scalar_one_or_none()
    return data_source_type


@pytest.mark.asyncio
async def test_add_data_source_with_defaults(session):
    service = DataSourceService(session)

    ds_type = await get_qh_sdmx21_ds_type(session)

    data_source = await service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source',
            type_id=ds_type.id,
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

    assert data_source.title == 'test_data_source'
    assert data_source.description == ''
    assert data_source.type_id == ds_type.id

    assert data_source.type.id == ds_type.id
    assert data_source.type.name == 'QH_SDMX21'

    assert data_source.details.get('description') == ''
    assert data_source.details.get('apiKeyHeader') == 'Ocp-Apim-Subscription-Key'
    assert data_source.details.get('apiKey') == 'test_key'
    assert data_source.details.get('locale') == 'en'

    assert data_source.details.get('sdmxConfig', {}).get('id') == 'test_id'
    assert data_source.details.get('sdmxConfig', {}).get('dataContentType') == 'JSON'
    assert data_source.details.get('sdmxConfig', {}).get('name') == 'test_name'
    assert data_source.details.get('sdmxConfig', {}).get('url') == 'https://example.com/sdmx.url'
    assert isinstance(data_source.details.get('sdmxConfig', {}).get('headers'), dict)
    assert len(data_source.details.get('sdmxConfig', {}).get('headers')) == 10
    assert isinstance(data_source.details.get('sdmxConfig', {}).get('supports'), dict)
    assert len(data_source.details.get('sdmxConfig', {}).get('supports')) == 11

    res = await service.get_schema_by_id(data_source.id)
    assert res == data_source


@pytest.mark.asyncio
async def test_add_data_source_with_custom_values(session):
    service = DataSourceService(session)

    ds_type = await get_qh_sdmx21_ds_type(session)

    data_source = await service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source_2',
            description='Other description',
            type_id=ds_type.id,
            details={
                'description': 'Testing description',
                'apiKey': 'test_api_key',
                'sdmxConfig': {
                    'id': 'test_002',
                    'name': 'Name 002',
                    'url': 'https://example.com/002',
                    'headers': {
                        "data": {"accept": "test/type"},
                        "dataflow": {"accept": "other/type"},
                    },
                    'supports': {"data": True, "dataflow": False},
                },
            },
        )
    )

    assert data_source.title == 'test_data_source_2'
    assert data_source.description == 'Other description'
    assert data_source.type_id == ds_type.id

    assert data_source.details.get('description') == 'Testing description'
    assert data_source.details.get('apiKey') == 'test_api_key'

    assert data_source.details.get('sdmxConfig', {}).get('id') == 'test_002'
    assert data_source.details.get('sdmxConfig', {}).get('name') == 'Name 002'
    assert data_source.details.get('sdmxConfig', {}).get('url') == 'https://example.com/002'

    target_headers = SdmxHeaders().model_dump(mode='json')  # Get default headers
    target_headers['data'] = {"accept": "test/type"}
    target_headers['dataflow'] = {"accept": "other/type"}
    assert data_source.details.get('sdmxConfig', {}).get('headers') == target_headers

    target_supports = SdmxSupport().model_dump(mode='json')  # Get default supports
    target_supports['data'] = True
    target_supports['dataflow'] = False

    assert data_source.details.get('sdmxConfig', {}).get('supports') == target_supports

    # Check if the data source really has this values in the database:
    res = await service.get_schema_by_id(data_source.id)
    assert res == data_source


@pytest.mark.asyncio
async def test_get_channel_list_and_count(session, clear_data_sources):
    service = DataSourceService(session)

    count = await service.get_data_sources_count()
    assert count == 0

    data_sources = await service.get_data_sources_schemas(limit=100, offset=0)
    assert len(data_sources) == 0

    data_source_1 = await service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source_3',
            type_id=1,
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

    data_source_2 = await service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source_4',
            type_id=1,
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

    count = await service.get_data_sources_count()
    assert count == 2

    data_sources = await service.get_data_sources_schemas(limit=100, offset=0)
    assert len(data_sources) == 2
    assert data_sources[0] == data_source_1
    assert data_sources[1] == data_source_2

    data_sources_filtered = await service.get_data_sources_schemas(
        limit=100, offset=0, ids=[data_source_2.id]
    )
    assert len(data_sources_filtered) == 1
    assert data_sources_filtered[0] == data_source_2


@pytest.mark.asyncio
async def test_update_data_source(session):
    service = DataSourceService(session)

    ds_type = await get_qh_sdmx21_ds_type(session)

    data_source = await service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source_5',
            type_id=ds_type.id,
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

    data_source_updated = await service.update(
        data_source.id,
        schemas.DataSourceUpdate(
            title='test_data_source_5_updated',
            description='Updated description',
            details={
                'apiKey': 'test_key_updated',
                'description': 'Updated test description',
                'sdmxConfig': {
                    'id': 'test_id_updated',
                    'name': 'test_name_updated',
                    'url': 'https://example.com/sdmx.url/updated',
                },
            },
        ),
    )

    assert data_source_updated.title == 'test_data_source_5_updated'
    assert data_source_updated.description == 'Updated description'
    assert data_source_updated.type_id == ds_type.id

    assert data_source_updated.details.get('description') == 'Updated test description'
    assert data_source_updated.details.get('apiKey') == 'test_key_updated'

    assert data_source_updated.details.get('sdmxConfig', {}).get('id') == 'test_id_updated'
    assert data_source_updated.details.get('sdmxConfig', {}).get('name') == 'test_name_updated'
    assert (
        data_source_updated.details.get('sdmxConfig', {}).get('url')
        == 'https://example.com/sdmx.url/updated'
    )

    # Check if the data source really has this values in the database:
    res = await service.get_schema_by_id(data_source.id)
    assert res == data_source_updated


@pytest.mark.asyncio
async def test_delete_data_source(session):
    service = DataSourceService(session)

    data_source = await service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source_6',
            type_id=1,
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

    res = await service.get_schema_by_id(data_source.id)
    assert res == data_source

    await service.delete(data_source.id)

    with pytest.raises(HTTPException) as e:
        await service.get_schema_by_id(data_source.id)

    assert e.value.status_code == 404

    data_sources = await service.get_data_sources_schemas(limit=100, offset=0)
    assert data_source.id not in (ds.id for ds in data_sources)

    audit_logs = await get_audit_logs(
        session,
        entity_type=schemas.AuditEntityType.DATA_SOURCE,
        action_type=schemas.AuditActionType.DELETE,
        item_ids=[data_source.id],
    )
    assert len(audit_logs) == 1
    assert audit_logs[0].state_after is None


@pytest.mark.asyncio
async def test_data_source_deletion_audits_related_datasets(session, clear_all, sdmx_clint_mock):
    data_source_service = DataSourceService(session)
    data_source_deletion_service = DataSourceDeletionService(session)
    dataset_service = DataSetService(session)

    data_source = await data_source_service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source_with_datasets',
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

    dataset_1 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=uuid.uuid4(),
            title='CPI Dataset 1',
            data_source_id=data_source.id,
            details={
                'urn': {'agencyId': 'IMF.STA', 'resourceId': 'CPI', 'version': '4.0.0'},
                'dimensions': {
                    'INDEX_TYPE': {'dimensionType': 'INDICATOR', 'isRequired': True},
                    'FREQUENCY': {'dimensionType': 'NON_INDICATOR', 'subtype': 'FREQUENCY'},
                    'TIME_PERIOD': {'dimensionType': 'TIME_PERIOD'},
                },
            },
        ),
        auth_context=SystemUserAuthContext(),
    )
    dataset_2 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=uuid.uuid4(),
            title='CPI Dataset 2',
            data_source_id=data_source.id,
            details={
                'urn': {'agencyId': 'IMF.STA', 'resourceId': 'CPI', 'version': '3.0.1'},
                'dimensions': {
                    'INDEX_TYPE': {'dimensionType': 'INDICATOR', 'isRequired': True},
                    'FREQUENCY': {'dimensionType': 'NON_INDICATOR', 'subtype': 'FREQUENCY'},
                    'TIME_PERIOD': {'dimensionType': 'TIME_PERIOD'},
                },
            },
        ),
        auth_context=SystemUserAuthContext(),
    )

    await data_source_deletion_service.delete(data_source.id)

    dataset_audit_logs = await get_audit_logs(
        session,
        entity_type=schemas.AuditEntityType.DATASET,
        action_type=schemas.AuditActionType.DELETE,
        item_ids=[dataset_1.id, dataset_2.id],
    )
    assert len(dataset_audit_logs) == 2
    assert all(log.state_after is None for log in dataset_audit_logs)

    data_source_audit_logs = await get_audit_logs(
        session,
        entity_type=schemas.AuditEntityType.DATA_SOURCE,
        action_type=schemas.AuditActionType.DELETE,
        item_ids=[data_source.id],
    )
    assert len(data_source_audit_logs) == 1
    assert data_source_audit_logs[0].state_after is None


@pytest.mark.asyncio
async def test_data_source_deletion_blocked_lists_all_in_use_datasets(
    session, clear_all, sdmx_clint_mock
):
    data_source_service = DataSourceService(session)
    data_source_deletion_service = DataSourceDeletionService(session)
    dataset_service = DataSetService(session)

    data_source = await data_source_service.create_data_source(
        schemas.DataSourceBase(
            title='test_data_source_blocked_delete',
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

    dataset_details = {
        'dimensions': {
            'INDEX_TYPE': {'dimensionType': 'INDICATOR', 'isRequired': True},
            'FREQUENCY': {'dimensionType': 'NON_INDICATOR', 'subtype': 'FREQUENCY'},
            'TIME_PERIOD': {'dimensionType': 'TIME_PERIOD'},
        },
    }
    dataset_1 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=uuid.uuid4(),
            title='CPI Dataset 1',
            data_source_id=data_source.id,
            details={'urn': {'agencyId': 'IMF.STA', 'resourceId': 'CPI', 'version': '4.0.0'}}
            | dataset_details,
        ),
        auth_context=SystemUserAuthContext(),
    )
    dataset_2 = await dataset_service.create_dataset(
        schemas.DataSetBase(
            id_=uuid.uuid4(),
            title='CPI Dataset 2',
            data_source_id=data_source.id,
            details={'urn': {'agencyId': 'IMF.STA', 'resourceId': 'CPI', 'version': '3.0.1'}}
            | dataset_details,
        ),
        auth_context=SystemUserAuthContext(),
    )

    # Both datasets are used by a channel, so the whole data source deletion is blocked.
    channel = await get_channel(session)
    await dataset_service.add_dataset_to_channel(channel.id, dataset_1.id)
    await dataset_service.add_dataset_to_channel(channel.id, dataset_2.id)

    with pytest.raises(DatasetInUseError) as exc_info:
        await data_source_deletion_service.delete(data_source.id)

    # All blocking datasets are reported, not just the first one.
    blocked_titles = {ds.dataset_title for ds in exc_info.value.blocking_datasets}
    assert blocked_titles == {'CPI Dataset 1', 'CPI Dataset 2'}

    # The failed deletion was rolled back: data source and both datasets still exist.
    remaining = await dataset_service.get_datasets_models(
        limit=None, offset=0, source_id=data_source.id
    )
    assert {ds.id for ds in remaining} == {dataset_1.id, dataset_2.id}
    assert await data_source_service.get_schema_by_id(data_source.id) is not None

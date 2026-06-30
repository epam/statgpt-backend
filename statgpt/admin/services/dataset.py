import asyncio
import json
import logging
import os.path
import uuid
import zipfile
from collections import Counter, defaultdict
from collections.abc import Generator, Iterable
from typing import Any, NamedTuple

import yaml
from fastapi import BackgroundTasks, HTTPException, status
from pydantic import ValidationError
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.expression import func, select

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.audit.decorators import audit_action
from statgpt.admin.audit.service import AuditService
from statgpt.admin.settings.exim import ExImSettings, JobsConfig
from statgpt.common import utils
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data import base
from statgpt.common.data.base.dataset import DataSetConfigType
from statgpt.common.hybrid_indexer import Indexer
from statgpt.common.schemas import (
    AuditActionType,
    AuditEntityType,
    AutoUpdateResult,
    ChannelIndexStatusScope,
    HybridSearchConfig,
)
from statgpt.common.schemas import PreprocessingStatusEnum as StatusEnum
from statgpt.common.services import (
    ChannelDataSetSerializer,
    ChannelSerializer,
    DataSetSerializer,
    DataSetService,
)
from statgpt.common.services.dataset import LastCompletedVersions
from statgpt.common.settings.document import (
    DimensionValueDocumentMetadataFields,
    IndicatorDocumentMetadataFields,
    SpecialDimensionValueDocumentMetadataFields,
)
from statgpt.common.utils import async_utils, crc32_hash_incremental_async, format_exception_reason
from statgpt.common.utils.elastic import ElasticIndex, ElasticSearchFactory, SearchResult
from statgpt.common.vectorstore import EmbeddinglessVectorStore, VectorStore, VectorStoreFactory

from .background_tasks import background_task
from .channel import AdminPortalChannelService as ChannelService
from .data_source import AdminPortalDataSourceService as DataSourceService
from .exceptions import BlockingDataset, DatasetInUseError
from .status_recovery import set_failed_status

_log = logging.getLogger(__name__)


class _DataHashes(NamedTuple):
    indicator_dimensions_hash: str
    non_indicator_dimensions_hash: str
    special_dimensions_hash: str | None


class _ReindexParams(NamedTuple):
    version_id: int
    channel_dataset_id: int
    harmonization_supported: bool
    status_on_completion: StatusEnum


class AutoUpdateChannelResult(NamedTuple):
    channel_id: int
    deployment_id: str
    total: int
    failed: int
    summary: str
    failed_reasons: list[str]


class AdminPortalDataSetService(DataSetService):

    def __init__(self, session: AsyncSession | None = None) -> None:
        super().__init__(session, None)

    _EXIM_SETTINGS = ExImSettings()

    @staticmethod
    def _get_elasticsearch_store_file_name(dataset: schemas.DataSet) -> str:
        return utils.escape_invalid_filename_chars(f"{dataset.title}.jsonl")

    async def _export_vector_store_data(
        self,
        channel: models.Channel,
        res_dir: str,
        auth_context: AuthContext,
        latest_completed_versions: dict[int, LastCompletedVersions],
    ) -> None:
        _log.info("Exporting vector store data...")
        vector_store_factory = VectorStoreFactory()

        # collect last completed version ids
        version_ids: set[int] = set()
        for versions in latest_completed_versions.values():
            if versions.last_completed_version:
                version_ids.add(versions.last_completed_version.version_data_id)

        _log.info(f"Exporting {len(version_ids)} version(s): {sorted(version_ids)}")

        collections = [
            channel.non_indicator_dimensions_table_name,
            channel.indicator_table_name,
            channel.special_dimensions_table_name,
        ]

        for table in collections:
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=table,
                auth_context=auth_context,
                embedding_model_name=channel.llm_model,
            )

            vector_store_folder = os.path.join(res_dir, table.split('_', maxsplit=1)[0])
            os.makedirs(vector_store_folder, exist_ok=True)

            await vector_store.export_to_folder(vector_store_folder, version_ids)

        _log.info("Finished exporting vector store data")

    @staticmethod
    async def _es_get_all(index: ElasticIndex, version_id: int) -> SearchResult:
        query = {"term": {"version_id": version_id}}
        res = await index.search(query=query, scroll="2m", size=10000)
        all_hits = res.hits.hits
        scroll_id = res.scroll_id
        while scroll_id is not None:
            res_scroll = await index.scroll(scroll_id=scroll_id, scroll="2m")
            hits = res_scroll.hits.hits
            if not hits:
                break
            all_hits.extend(hits)
            scroll_id = res_scroll.scroll_id
        res.hits.hits = all_hits
        return res

    async def _export_single_dataset_elastic_data(
        self,
        index: ElasticIndex,
        dataset: schemas.DataSet,
        version_id: int,
        index_folder: str,
    ) -> None:
        _log.info(f"Exporting elastic data (dataset: {dataset.title})...")
        res = await self._es_get_all(index, version_id=version_id)
        documents = [hit.source for hit in res.hits.hits]

        file_name = self._get_elasticsearch_store_file_name(dataset)
        file_path = os.path.join(index_folder, file_name)

        for d in documents:
            utils.write_json(
                obj=d,
                fp=file_path,
                mode='a+',
                encoding=JobsConfig.ENCODING,
                indent=None,
                add_newline=True,
            )
        _log.info(f"Exported elastic data (dataset: {dataset.title})")

    async def _export_elastic_data(
        self,
        channel: models.Channel,
        datasets: Iterable[schemas.DataSet],
        latest_completed_versions: dict[int, LastCompletedVersions],
        res_dir: str,
    ) -> None:

        _log.info("Exporting elastic data...")

        matching_index = await ElasticSearchFactory.get_index(channel.matching_index_name)
        indicators_index = await ElasticSearchFactory.get_index(channel.indicators_index_name)

        indexes = [
            (JobsConfig.ES_MATCHING_DIR, matching_index),
            (JobsConfig.ES_INDICATORS_DIR, indicators_index),
        ]

        for folder, index in indexes:
            index_folder = os.path.join(res_dir, folder)
            os.makedirs(index_folder, exist_ok=True)

            tasks = []
            for dataset in datasets:
                versions = latest_completed_versions[dataset.id]
                if not versions.last_completed_version:
                    _log.warning(f"Dataset '{dataset.title}' has no completed versions. Skipping.")
                    continue
                version_id = versions.last_completed_version.version_data_id
                tasks.append(
                    self._export_single_dataset_elastic_data(
                        index, dataset, version_id, index_folder
                    )
                )

            await async_utils.gather_with_concurrency(
                self._EXIM_SETTINGS.elastic_concurrency_limit, *tasks
            )
        _log.info("Finished exporting elastic data")

    @staticmethod
    def _export_datasets_config(
        datasets: list[schemas.DataSet], res_dir: str
    ) -> dict[int, schemas.DataSource]:
        data_sources = {}
        data = []
        for dataset in datasets:
            dataset_json = dataset.model_dump(mode='json', include=JobsConfig.DATASET_FIELDS)
            if dataset.data_source is None:
                raise ValueError("Dataset data_source is not loaded")
            dataset_json['dataSource'] = dataset.data_source.title
            data.append(dataset_json)

            if dataset.data_source_id not in data_sources:
                data_sources[dataset.data_source_id] = dataset.data_source

        data.sort(key=lambda x: x['title'])

        datasets_file = os.path.join(res_dir, JobsConfig.DATASETS_FILE)
        utils.write_yaml({'dataSets': data}, datasets_file)

        return data_sources

    @staticmethod
    def _export_versions(
        versions: dict[uuid.UUID, schemas.ChannelDatasetVersion], res_dir: str
    ) -> None:
        data = {
            str(ds_id): version.model_dump(mode='json', include=JobsConfig.VERSIONS_FIELDS)
            for ds_id, version in versions.items()
        }
        datasets_file = os.path.join(res_dir, JobsConfig.VERSIONS_FILE)
        utils.write_yaml({'data': data}, datasets_file)

    async def export_datasets(
        self,
        channel: models.Channel,
        res_dir: str,
        scope: schemas.ExportScope,
        auth_context: AuthContext,
    ) -> None:
        channel_config = schemas.ChannelConfig.model_validate(channel.details)

        datasets = await self.get_datasets_schemas(
            limit=None,
            offset=0,
            channel_id=channel.id,
            auth_context=auth_context,
            allow_offline=True,
        )

        if scope.includes_configs():
            data_sources = self._export_datasets_config(datasets, res_dir)
            await DataSourceService.export_data_sources(data_sources.values(), res_dir)

        if not scope.includes_indexes():
            return

        channel_datasets = await self.get_channel_dataset_models(
            limit=None, offset=0, channel_id=channel.id
        )
        latest_completed_versions = await self._get_latest_successful_dataset_version(
            channel_dataset_ids=[cd.id for cd in channel_datasets]
        )

        versions = {
            next(d.id_ for d in datasets if d.id == ds_id): version.last_completed_version
            for ds_id, version in latest_completed_versions.items()
            if version.last_completed_version is not None
        }
        self._export_versions(versions, res_dir)

        await self._export_vector_store_data(
            channel, res_dir, auth_context, latest_completed_versions
        )

        if channel_config.data_query is None:
            _log.info("No data query configured, skipping elastic data export")
            return

        indexer_version = channel_config.data_query.details.indexer_version
        _log.info(f"Indexer version: {indexer_version}")
        if indexer_version == schemas.IndexerVersion.hybrid:
            await self._export_elastic_data(channel, datasets, latest_completed_versions, res_dir)
        else:
            _log.info("Skipping exporting elastic data")

    async def _import_datasets(
        self,
        zip_file: zipfile.ZipFile,
        data_sources: dict[str, models.DataSource],
        update_datasets: bool,
        auth_context: AuthContext,
    ) -> list[schemas.DataSet]:
        existing_datasets = {
            ds.id_: ds
            for ds in await self.get_datasets_schemas(
                limit=None, offset=0, allow_offline=True, auth_context=auth_context
            )
        }

        datasets = []
        with zip_file.open(JobsConfig.DATASETS_FILE) as file:
            data = yaml.safe_load(file)

            for dataset_cfg in data['dataSets']:
                data_source = data_sources[dataset_cfg.pop('dataSource')]
                dataset_cfg["data_source_id"] = data_source.id
                parsed_dataset = schemas.DataSetBase.model_validate(dataset_cfg)

                if dataset := existing_datasets.get(parsed_dataset.id_):
                    if update_datasets:
                        data = {
                            field: getattr(parsed_dataset, field)
                            for field in schemas.DataSetUpdateRequest.model_fields.keys()
                            if getattr(parsed_dataset, field) != getattr(dataset, field)
                        }
                        if data:
                            _log.info(f"Updating dataset '{dataset_cfg['title']}' with {data}")
                            update_response = await self.update(
                                dataset.id,
                                schemas.DataSetUpdateRequest.model_validate(data),
                                auth_context=auth_context,
                            )
                            dataset = update_response.dataset
                        else:
                            _log.info(f"Dataset '{dataset_cfg['title']}' exists and is up to date")
                    else:
                        _log.info(f"Dataset '{dataset_cfg['title']}' already exists. Skipping.")
                else:
                    dataset = await self.create_dataset(parsed_dataset, auth_context=auth_context)
                    dataset.data_source = data_source  # type: ignore
                    _log.info(f"Created dataset {dataset.title}")

                datasets.append(dataset)

        return datasets

    async def _add_datasets_to_channel(
        self, channel_id: int, datasets: list[schemas.DataSet]
    ) -> None:
        items = [
            models.ChannelDataset(
                channel_id=channel_id,
                dataset_id=ds.id,
            )
            for ds in datasets
        ]

        self._session.add_all(items)
        await self._session.commit()

    async def _import_datasets_versions(
        self, zip_file: zipfile.ZipFile, datasets: list[schemas.DataSet], channel_id: int
    ) -> dict[int, models.ChannelDatasetVersion]:
        with zip_file.open(JobsConfig.VERSIONS_FILE) as file:
            versions_json = yaml.safe_load(file)

        datasets_dict = {ds.id: ds for ds in datasets}

        channel_datasets = await self.get_channel_dataset_models(
            limit=None, offset=0, channel_id=channel_id
        )
        versions = {}
        for ch_ds in channel_datasets:
            dataset = datasets_dict[ch_ds.dataset_id]

            other = {}
            if v := versions_json['data'].get(str(dataset.id_)):
                other['creation_reason'] = "Imported from zip"
                other.update(v)
            else:
                _log.warning(f"No version data found for dataset {dataset.title!r}")
                other['creation_reason'] = "Imported from zip without version data"
            version = models.ChannelDatasetVersion(
                channel_dataset_id=ch_ds.id,
                # `version` will be set by the DB trigger automatically
                preprocessing_status=StatusEnum.IN_PROGRESS,
                **other,
            )
            versions[ch_ds.dataset_id] = version
        self._session.add_all(versions.values())
        await self._session.commit()
        return versions

    async def _import_vector_store_tables(
        self,
        zip_file: zipfile.ZipFile,
        channel: models.Channel,
        datasets: list[schemas.DataSet],
        versions: dict[int, models.ChannelDatasetVersion],
        auth_context: AuthContext,
    ) -> None:
        _log.info("Importing vector store data...")
        vector_store_factory = VectorStoreFactory()

        dataset_versions: dict[uuid.UUID, int] = {
            dataset.id_: versions[dataset.id].id for dataset in datasets
        }
        data_sources: dict[uuid.UUID, int] = {
            dataset.id_: dataset.data_source_id for dataset in datasets
        }

        collections = [
            channel.non_indicator_dimensions_table_name,
            channel.indicator_table_name,
            channel.special_dimensions_table_name,
        ]

        for table in collections:
            table_folder = table.split('_', maxsplit=1)[0]

            vector_store = await vector_store_factory.get_vector_store(
                collection_name=table,
                embedding_model_name=channel.llm_model,
                auth_context=auth_context,
            )

            await vector_store.import_from_zipfile(
                zip_file, table_folder, dataset_versions, data_sources
            )

        _log.info("Finished importing vector store data")
        _log.info('-' * 40)

    async def _import_elastic_data(
        self,
        zip_file: zipfile.ZipFile,
        channel: models.Channel,
        datasets: list[schemas.DataSet],
        versions: dict[int, models.ChannelDatasetVersion],
    ) -> None:
        _log.info("Importing elastic data...")

        matching_index = await ElasticSearchFactory.get_index(
            channel.matching_index_name, allow_creation=True
        )
        indicators_index = await ElasticSearchFactory.get_index(
            channel.indicators_index_name, allow_creation=True
        )

        indexes = [
            (JobsConfig.ES_MATCHING_DIR, matching_index),
            (JobsConfig.ES_INDICATORS_DIR, indicators_index),
        ]

        for folder, index in indexes:
            for dataset in datasets:
                version = versions[dataset.id]

                file_name = self._get_elasticsearch_store_file_name(dataset)
                file_path = f"{folder}/{file_name}"

                if file_path not in zip_file.namelist():
                    _log.warning(f"File '{file_path}' not found in the zip archive")
                    continue
                _log.info(f"Opening '{file_path}'")

                with zip_file.open(file_path) as file:
                    documents: list[dict[str, str]] = []
                    for line in file.readlines():
                        doc = json.loads(line)

                        # Ensure dataset metadata is valid:
                        doc['dataset_id'] = str(dataset.id_)
                        doc['dataset_name'] = dataset.title
                        doc['version_id'] = version.id

                        documents.append(doc)

                await index.add_bulk(documents)
        _log.info("Finished importing elastic data")
        _log.info('-' * 40)

    async def import_datasets_and_data_sources_from_zip(
        self,
        channel_db: models.Channel,
        zip_file: zipfile.ZipFile,
        update_datasets: bool,
        update_data_sources: bool,
        scope: schemas.ExportScope,
        auth_context: AuthContext,
    ) -> None:
        datasets = await self._import_or_load_datasets(
            zip_file, channel_db, update_datasets, update_data_sources, scope, auth_context
        )

        if scope.includes_indexes():
            channel_config = schemas.ChannelConfig.model_validate(channel_db.details)
            versions = await self._import_datasets_versions(
                zip_file, datasets, channel_id=channel_db.id
            )
            await self._import_indexes(
                zip_file, channel_db, datasets, versions, channel_config, auth_context
            )
            await self._mark_versions_completed(versions)

        await self._session.commit()

    async def _import_or_load_datasets(
        self,
        zip_file: zipfile.ZipFile,
        channel_db: models.Channel,
        update_datasets: bool,
        update_data_sources: bool,
        scope: schemas.ExportScope,
        auth_context: AuthContext,
    ) -> list[schemas.DataSet]:
        if scope.includes_configs():
            source_service = DataSourceService(self._session)
            data_sources = await source_service.import_data_sources_from_zip(
                zip_file, update_data_sources
            )
            datasets = await self._import_datasets(
                zip_file, data_sources, update_datasets, auth_context=auth_context  # type: ignore
            )
            await self._add_datasets_to_channel(channel_id=channel_db.id, datasets=datasets)
            return datasets

        return await self.get_datasets_schemas(
            limit=None,
            offset=0,
            channel_id=channel_db.id,
            auth_context=auth_context,
            allow_offline=True,
        )

    async def _import_indexes(
        self,
        zip_file: zipfile.ZipFile,
        channel_db: models.Channel,
        datasets: list[schemas.DataSet],
        versions: dict[int, models.ChannelDatasetVersion],
        channel_config: schemas.ChannelConfig,
        auth_context: AuthContext,
    ) -> None:
        await self._import_vector_store_tables(
            zip_file, channel_db, datasets, versions, auth_context
        )
        await self._import_elastic_data_if_needed(
            zip_file, channel_db, datasets, versions, channel_config
        )

    async def _import_elastic_data_if_needed(
        self,
        zip_file: zipfile.ZipFile,
        channel_db: models.Channel,
        datasets: list[schemas.DataSet],
        versions: dict[int, models.ChannelDatasetVersion],
        channel_config: schemas.ChannelConfig,
    ) -> None:
        if channel_config.data_query is None:
            _log.info("No data query configured, skipping data import")
            return

        indexer_version = channel_config.data_query.details.indexer_version
        _log.info(f"Indexer version: {indexer_version}")

        if indexer_version == schemas.IndexerVersion.hybrid:
            await self._import_elastic_data(zip_file, channel_db, datasets, versions)
        else:
            _log.info("Skipping importing elastic data")

    async def _mark_versions_completed(
        self, versions: dict[int, models.ChannelDatasetVersion]
    ) -> None:
        for item in versions.values():
            await self._update_channel_dataset_version_status(
                item, new_status=StatusEnum.COMPLETED, do_commit=False
            )

    async def _parse_details_field(
        self, handler: base.DataSourceHandler, details: dict[str, Any]
    ) -> DataSetConfigType:
        try:
            parsed_config = handler.parse_data_set_config(details)
        except ValidationError as e:
            _log.info(e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse 'details' field: {e}",
            )
        except Exception as e:
            _log.info(e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to parse 'details' field",
            )
        return parsed_config

    @audit_action(entity_type=AuditEntityType.DATASET, action_type=AuditActionType.CREATE)
    async def create_dataset(
        self, data: schemas.DataSetBase, auth_context: AuthContext
    ) -> schemas.DataSet:
        handler = await self._get_handler(data.data_source_id)
        dataset_config = await self._parse_details_field(handler, data.details)  # type: ignore

        item = models.DataSet(
            id_=data.id_,
            title=data.title,
            source_id=data.data_source_id,
            details=dataset_config.model_dump(mode='json', by_alias=True),
        )

        self._session.add(item)
        await self._session.flush()

        dataset = await handler.get_dataset(
            entity_id=item.id_,
            title=item.title,
            config=item.details,
            auth_context=auth_context,
            allow_offline=True,
        )
        return DataSetSerializer.db_to_schema(item, dataset)

    async def load_available_datasets(
        self,
        source_id: int,
        auth_context: AuthContext,
        *,
        provider: str | None = None,
    ) -> list[schemas.DataSetDescriptor]:
        handler = await self._get_handler(source_id)

        datasets = []
        for ds in await handler.list_datasets(auth_context, provider=provider):
            datasets.append(
                schemas.DataSetDescriptor(
                    data_source_id=source_id,
                    id_in_source=ds.id_in_source,
                    title=ds.name,
                    description=ds.description or "",
                    details=ds.details.model_dump(mode="json", by_alias=True),
                )
            )

        return datasets

    async def load_available_providers(
        self, source_id: int, auth_context: AuthContext
    ) -> list[schemas.Provider]:
        handler = await self._get_handler(source_id)
        return await handler.list_providers(auth_context)

    async def get_dataset_config_schema(self, source_id: int) -> dict:
        """Returns JSON schema for dataset configuration."""
        handler = await self._get_handler(source_id)
        return handler.get_data_set_config_schema()

    async def validate_config(
        self, source_id: int, config: dict, auth_context: AuthContext
    ) -> base.DataSetValidationResult:
        handler = await self._get_handler(source_id)
        res = await handler.validate_dataset_config(
            config, auth_context=auth_context, mode="return"
        )
        return res

    async def get_dataset_structure(
        self, source_id: int, config: dict, auth_context: AuthContext
    ) -> dict:
        handler = await self._get_handler(source_id)
        structure = await handler.get_dataset_structure(config, auth_context=auth_context)
        return structure.model_dump(mode='json', by_alias=True)

    @audit_action(entity_type=AuditEntityType.DATASET, action_type=AuditActionType.UPDATE)
    async def update(
        self, item_id: int, data: schemas.DataSetUpdateRequest, auth_context: AuthContext
    ) -> schemas.DataSetUpdateResponse:
        item = await self.get_model_by_id(item_id, expand=True)

        for attr, value in data.model_dump(exclude_unset=True, exclude={'details'}).items():
            setattr(item, attr, value)

        handler = await self._get_handler(item.source_id)
        if data.details is not None:
            dataset_config = await self._parse_details_field(handler, data.details)  # type: ignore
            item.details = dataset_config.model_dump(mode='json', by_alias=True)

        item.updated_at = func.now()
        await self._session.flush()
        await self._session.refresh(item)

        dataset = await handler.get_dataset(
            entity_id=item.id_,
            title=item.title,
            config=item.details,
            auth_context=auth_context,
            allow_offline=True,
        )
        dataset_schema = DataSetSerializer.db_to_schema(item, dataset, expand=True)

        channel_results = await self._propagate_config_to_channel_datasets(
            item, handler, auth_context=auth_context
        )
        return schemas.DataSetUpdateResponse(
            dataset=dataset_schema,
            channel_results=channel_results,
        )

    async def _validate_datasets_deletable(self, items: list[models.DataSet]) -> None:
        if not items:
            return

        dataset_ids = [item.id for item in items]
        query = (
            select(models.ChannelDataset.dataset_id, func.count())
            .where(models.ChannelDataset.dataset_id.in_(dataset_ids))
            .group_by(models.ChannelDataset.dataset_id)
        )
        in_use = (await self._session.execute(query)).all()
        if not in_use:
            return

        datasets_by_id = {item.id: item for item in items}
        blocking = sorted(
            (
                BlockingDataset(
                    dataset_id=dataset_id,
                    dataset_title=datasets_by_id[dataset_id].title,
                    channel_count=count,
                )
                for dataset_id, count in in_use
            ),
            key=lambda ds: ds.dataset_title,
        )
        details = ", ".join(f"id={ds.dataset_id} ({ds.channels_label})" for ds in blocking)
        _log.warning(f"Cannot delete dataset(s) used in channels: {details}")
        raise DatasetInUseError(blocking)

    @staticmethod
    def _deleted_schema_from_model(item: models.DataSet) -> schemas.DeletedDataSet:
        return schemas.DeletedDataSet(
            id=item.id,
            id_=item.id_,
            title=item.title,
            data_source_id=item.source_id,
            details=item.details,
        )

    async def _delete_datasets(self, datasets: Iterable[models.DataSet]) -> None:
        items = list(datasets)
        if not items:
            return

        async def _do_delete() -> None:
            await self._validate_datasets_deletable(items)
            for item in items:
                _log.info(f"Deleting dataset(id={item.id}): {item.title!r}")

            deleted = [self._deleted_schema_from_model(item) for item in items]
            await self._session.execute(
                delete(models.DataSet).where(models.DataSet.id.in_([item.id for item in items]))
            )
            await self._session.flush()
            AuditService(self._session).persist_batch(
                entity_type=AuditEntityType.DATASET,
                action_type=AuditActionType.DELETE,
                items=deleted,
            )

        if self._session.in_transaction():
            await _do_delete()
            return

        async with self._session.begin():
            await _do_delete()

    async def delete_datasets_by_source_id(self, source_id: int) -> None:
        datasets = await self.get_datasets_models(
            limit=None,
            offset=0,
            source_id=source_id,
        )
        await self._delete_datasets(datasets)

    @audit_action(entity_type=AuditEntityType.DATASET, action_type=AuditActionType.DELETE)
    async def delete(self, item_id: int) -> schemas.DeletedDataSet:
        item = await self.get_model_by_id(item_id)
        await self._validate_datasets_deletable([item])

        _log.info(f"Deleting dataset(id={item.id}): {item.title!r}")
        deleted_item = self._deleted_schema_from_model(item)
        await self._session.delete(item)
        await self._session.flush()
        return deleted_item

    async def _propagate_config_to_channel_datasets(
        self,
        dataset: models.DataSet,
        handler: base.DataSourceHandler,
        auth_context: AuthContext,
    ) -> list[schemas.ChannelDatasetUpdateResult]:
        """Propagate config changes to all channel datasets that reference this dataset.

        For each channel dataset:
        - If indexing is in progress -> INDEXING_IN_PROGRESS
        - If no completed version exists -> NO_VERSION
        - If the resolved URN differs from last_completed, or structure hash differs,
          or indexing config hash differs -> NEEDS_REINDEX
        - Otherwise -> AUTO_UPDATED (creates a new pointer version)
        """
        await self._session.refresh(dataset, attribute_names=["mapped_channels"])

        if not dataset.mapped_channels:
            _log.info(f"Dataset(id={dataset.id_}): no mapped channels, skipping config propagation")
            return []

        results: list[schemas.ChannelDatasetUpdateResult] = []

        channel_dataset_ids = [cd.id for cd in dataset.mapped_channels]

        latest_versions = await self._get_latest_channel_dataset_versions(channel_dataset_ids)
        last_completed_mapping = await self._get_latest_successful_channel_dataset_versions(
            channel_dataset_ids=channel_dataset_ids
        )

        for channel_dataset in dataset.mapped_channels:
            await self._session.refresh(channel_dataset, attribute_names=["channel"])
            channel = channel_dataset.channel

            latest_version = latest_versions.get(channel_dataset.id)
            last_completed_versions = last_completed_mapping.get(channel_dataset.id)
            last_completed = (
                last_completed_versions.last_completed_version if last_completed_versions else None
            )

            other_fields = {}
            if (
                latest_version
                and latest_version.preprocessing_status not in StatusEnum.final_statuses()
            ):
                status = schemas.ChannelDatasetUpdateStatus.INDEXING_IN_PROGRESS
                _log.info(
                    f"ChannelDataset(dataset={dataset.id_}, channel={channel.deployment_id!r}):"
                    f" indexing in progress, skipping"
                )
            elif last_completed is None:
                status = schemas.ChannelDatasetUpdateStatus.NO_VERSION
                _log.info(
                    f"ChannelDataset(dataset={dataset.id_}, channel={channel.deployment_id!r}):"
                    f" no completed version, skipping"
                )
            else:
                status, new_resolved_config, reasons = await self._classify_config_update(
                    handler=handler,
                    current_config=dataset.details,
                    last_completed=last_completed,
                    auth_context=auth_context,
                )
                if status is schemas.ChannelDatasetUpdateStatus.AUTO_UPDATED:
                    new_version = await self._apply_config_internal(
                        channel_dataset,
                        last_completed,
                        handler,
                        current_config=dataset.details,
                        resolved_config=new_resolved_config,
                    )
                    other_fields['new_version'] = new_version
                    _log.info(
                        f"ChannelDataset(dataset={dataset.id_}, channel={channel.deployment_id!r}):"
                        f" auto-updated with new version"
                    )
                else:
                    _log.info(
                        f"ChannelDataset(dataset={dataset.id_}, channel={channel.deployment_id!r}):"
                        f" needs reindex ({', '.join(reasons)})"
                    )

            results.append(
                schemas.ChannelDatasetUpdateResult(
                    channel_dataset_id=channel_dataset.id,
                    status=status,
                    channel=ChannelSerializer.db_to_schema(channel),
                    **other_fields,
                )
            )

        return results

    @staticmethod
    async def _classify_config_update(
        handler: base.DataSourceHandler,
        current_config: dict[str, Any],
        last_completed: schemas.ChannelDatasetVersion,
        auth_context: AuthContext,
    ) -> tuple[schemas.ChannelDatasetUpdateStatus, dict[str, Any], list[str]]:
        """Determine whether a dataset config update requires re-indexing.

        Returns:
            A tuple of (status, new_resolved_config, reasons).
            - status: AUTO_UPDATED when no reindex is needed; NEEDS_REINDEX otherwise.
            - new_resolved_config: the (re)resolved config (always populated).
            - reasons: human-readable triggers when reindex is required; empty otherwise.
        """
        if last_completed.resolved_config is None:
            _, new_resolved_config = await handler.resolve_config(
                config=current_config, auth_context=auth_context
            )
            urn_changed = False
        else:
            _, new_resolved_config = await handler.reresolve_config(
                config=current_config,
                previous_resolved_config=last_completed.resolved_config,
                auth_context=auth_context,
            )
            urn_changed = new_resolved_config != last_completed.resolved_config

        new_config_hash = handler.parse_data_set_config(new_resolved_config).indexing_hash
        new_structure_hash, _ = await handler.get_structure_hash_and_metadata(
            dataset_config=new_resolved_config, auth_context=auth_context
        )

        reasons: list[str] = []
        if urn_changed:
            reasons.append("URN changed")
        if new_structure_hash != last_completed.structure_hash:
            reasons.append("structure hash changed")
        if new_config_hash != last_completed.indexing_config_hash:
            reasons.append("indexing config hash changed")

        status = (
            schemas.ChannelDatasetUpdateStatus.NEEDS_REINDEX
            if reasons
            else schemas.ChannelDatasetUpdateStatus.AUTO_UPDATED
        )
        return status, new_resolved_config, reasons

    async def _apply_config_internal(
        self,
        channel_dataset: models.ChannelDataset,
        last_completed: schemas.ChannelDatasetVersion,
        handler: base.DataSourceHandler,
        current_config: dict[str, Any],
        resolved_config: dict[str, Any],
    ) -> schemas.ChannelDatasetVersion:
        """Apply config changes to a channel dataset without re-indexing.

        Creates a new COMPLETED version that reuses indexed data from the
        last completed version.
        """
        if last_completed.resolved_config is None:
            new_resolved_config = dict(current_config)
        else:
            new_resolved_config = handler.merge_config_with_resolved(
                current_config=current_config,
                resolved_config=last_completed.resolved_config,
            )
        # URN is no longer an IndexingField, so the merge above takes URN from
        # current_config (potentially "latest"). Restore the resolved URN to keep
        # the resolved_config invariant.
        new_resolved_config['urn'] = resolved_config['urn']

        parsed_config = handler.parse_data_set_config(new_resolved_config)

        new_item = models.ChannelDatasetVersion(
            channel_dataset_id=channel_dataset.id,
            # `version` will be set by the DB trigger automatically
            preprocessing_status=StatusEnum.COMPLETED,
            pointer_to=last_completed.version_data_id,
            creation_reason="Applied dataset config changes without re-indexing",
            resolved_config=new_resolved_config,
            indexing_config_hash=parsed_config.indexing_hash,
            structure_metadata=last_completed.structure_metadata,
            structure_hash=last_completed.structure_hash,
            indicator_dimensions_hash=last_completed.indicator_dimensions_hash,
            non_indicator_dimensions_hash=last_completed.non_indicator_dimensions_hash,
            special_dimensions_hash=last_completed.special_dimensions_hash,
        )

        self._session.add(new_item)
        await self._session.flush()
        await self._session.refresh(new_item)
        return schemas.ChannelDatasetVersion.model_validate(new_item, from_attributes=True)

    async def add_dataset_to_channel(
        self, channel_id: int, dataset_id: int
    ) -> schemas.ChannelDatasetBase:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        dataset: models.DataSet = await self.get_model_by_id(dataset_id)

        if await self.get_channel_dataset_model_or_none(
            channel_id=channel_id, dataset_id=dataset_id
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The dataset has already been added to the channel",
            )

        item = models.ChannelDataset(
            channel_id=channel.id,
            dataset_id=dataset.id,
        )

        self._session.add(item)
        await self._session.commit()

        return schemas.ChannelDatasetBase.model_validate(item, from_attributes=True)

    async def remove_channel_dataset(self, channel_id: int, dataset_id: int) -> None:
        # Phase A: DB reads
        async with self._scoped_session():
            channel: schemas.Channel = await ChannelService(self._session).get_schema_by_id(
                channel_id
            )
            dataset: models.DataSet = await self.get_model_by_id(dataset_id)
            channel_dataset = await self.get_channel_dataset_model_or_none(
                channel_id=channel.id, dataset_id=dataset.id
            )
            if not channel_dataset:
                return
            dataset_uuid = dataset.id_
            channel_dataset_id = channel_dataset.id

        # Phase B: Clear data — no DB session held
        await self._clear_channel_dataset_data(channel, dataset_id=dataset_uuid, version_ids=None)

        # Phase C: DB write — delete channel_dataset
        async with self._scoped_session():
            channel_dataset = await self._get_channel_dataset_model_or_raise(channel_dataset_id)
            await self._session.delete(channel_dataset)
            await self._session.commit()

    async def _clear_channel_dataset_data(
        self,
        channel: schemas.Channel,
        *,
        dataset_id: uuid.UUID | None,
        version_ids: list[int] | None,
    ) -> None:
        """Clears all data related to a dataset or specific versions of a dataset"""
        await self._clear_vector_stores(channel, dataset_id=dataset_id, version_ids=version_ids)
        if ChannelService.is_channel_hybrid(channel):
            await self._clear_elastic_indices(
                channel, dataset_id=dataset_id, version_ids=version_ids
            )

    async def _clear_vector_stores(
        self,
        channel: schemas.Channel,
        dataset_id: uuid.UUID | None,
        version_ids: list[int] | None,
    ) -> None:
        vector_store_factory = VectorStoreFactory()

        collections = [
            channel.indicator_table_name,
            channel.special_dimensions_table_name,
            channel.non_indicator_dimensions_table_name,
        ]
        for col in collections:
            vector_store = await vector_store_factory.get_embeddingless_vector_store(col)
            await vector_store.remove_documents_by(dataset_id=dataset_id, version_ids=version_ids)

    @staticmethod
    async def _clear_elastic_indices(
        channel: schemas.Channel, *, dataset_id: uuid.UUID | None, version_ids: list[int] | None
    ) -> None:
        _log.info("[Elastic] Clearing existing indicators in the matching and indicators indexes")

        matching_index = await ElasticSearchFactory.get_index(
            channel.matching_index_name, allow_creation=True
        )
        indicators_index = await ElasticSearchFactory.get_index(
            channel.indicators_index_name, allow_creation=True
        )
        if dataset_id and version_ids:
            raise ValueError("Provide either dataset_id or version_ids, not both")
        elif dataset_id:
            query: dict = {"bool": {"must": [{"term": {"dataset_id.keyword": str(dataset_id)}}]}}
        elif version_ids:
            query = {"bool": {"must": [{"terms": {"version_id": version_ids}}]}}
        else:
            raise ValueError("Either dataset_id or version_ids must be provided")
        _log.debug(f"[Elastic] Deleting documents with query: {query}")

        res1 = await matching_index.delete_by_query(query=query)
        res2 = await indicators_index.delete_by_query(query=query)

        _log.info(f"[Elastic] Matching index cleared: {res1}")
        _log.info(f"[Elastic] Indicators index cleared: {res2}")

    async def _create_new_channel_dataset_version(
        self,
        channel_dataset_id: int,
        reason: str,
        preprocessing_status: StatusEnum,
        resolved_config: dict | None = None,
    ) -> models.ChannelDatasetVersion:
        item = models.ChannelDatasetVersion(
            channel_dataset_id=channel_dataset_id,
            # `version` will be set by the DB trigger automatically
            preprocessing_status=preprocessing_status,
            creation_reason=reason,
        )
        if resolved_config is not None:
            item.resolved_config = resolved_config

        self._session.add(item)
        await self._session.commit()
        await self._session.refresh(item)
        return item

    async def _update_channel_dataset_status(
        self, item: models.ChannelDataset, new_status: StatusEnum, do_commit: bool = True
    ) -> None:
        item.clearing_status = new_status
        item.updated_at = func.now()

        if do_commit:
            await self._session.commit()
            await self._session.refresh(item)

    async def _update_channel_dataset_version_status(
        self,
        item: models.ChannelDatasetVersion,
        new_status: StatusEnum,
        reason_for_failure: str | None = None,
        do_commit: bool = True,
    ) -> None:
        item.preprocessing_status = new_status
        item.updated_at = func.now()

        if reason_for_failure:
            item.reason_for_failure = reason_for_failure

        if do_commit:
            await self._session.commit()
            await self._session.refresh(item)

    async def _set_version_hashes_and_metadata(
        self,
        item: models.ChannelDatasetVersion,
        config_hash: str,
        structure_hash: str,
        structure_metadata: dict,
        data_hashes: _DataHashes,
    ) -> None:
        """Sets the structure and data hashes for the given channel dataset version."""
        item.indexing_config_hash = config_hash
        item.structure_metadata = structure_metadata
        item.structure_hash = structure_hash
        item.indicator_dimensions_hash = data_hashes.indicator_dimensions_hash
        item.non_indicator_dimensions_hash = data_hashes.non_indicator_dimensions_hash
        item.special_dimensions_hash = data_hashes.special_dimensions_hash
        item.updated_at = func.now()

        await self._session.commit()
        await self._session.refresh(item)

    async def _set_resolved_config(
        self,
        item: models.ChannelDatasetVersion,
        resolved_config: dict | None,
    ) -> None:
        """Sets the resolved configuration for the given channel dataset version."""
        if resolved_config is not None:
            item.resolved_config = resolved_config
            item.updated_at = func.now()
            await self._session.commit()
            await self._session.refresh(item)

    async def _is_indexing_in_progress(self, channel_dataset_id: int) -> bool:
        """Checks if indexing is currently in progress for the given channel dataset."""
        latest_version = await self._get_latest_channel_dataset_version_model(
            channel_dataset_id=channel_dataset_id
        )
        return (
            latest_version is not None
            and latest_version.preprocessing_status not in StatusEnum.final_statuses()
        )

    async def rollback_channel_dataset_to_previous_version(
        self, channel_id: int, dataset_id: int
    ) -> schemas.ChannelDatasetVersion:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        dataset: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel.id, dataset_id=dataset.id
        )

        if await self._is_indexing_in_progress(channel_dataset.id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot roll back while the indexing is in progress.",
            )

        last_completed_versions = await self._get_latest_successful_channel_dataset_versions(
            channel_dataset_ids=[channel_dataset.id]
        )
        previous_version = last_completed_versions[channel_dataset.id].previous_completed_version
        if previous_version is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No previous completed version available to roll back to.",
            )

        new_item = models.ChannelDatasetVersion(
            channel_dataset_id=channel_dataset.id,
            # `version` will be set by the DB trigger automatically
            preprocessing_status=StatusEnum.COMPLETED,
            pointer_to=previous_version.version_data_id,
            creation_reason=f"Rolled back to previous version={previous_version.version}",
            **{f: getattr(previous_version, f) for f in JobsConfig.VERSIONS_FIELDS},
        )
        self._session.add(new_item)
        await self._session.commit()
        await self._session.refresh(new_item)
        return schemas.ChannelDatasetVersion.model_validate(new_item, from_attributes=True)

    async def is_channel_dataset_latest_version_up_to_date(
        self, channel_id: int, dataset_id: int, auth_context: AuthContext
    ) -> schemas.ChangesBetweenVersionAndActualData:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        dataset_db: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel.id, dataset_id=dataset_db.id
        )

        latest_completed_version = await self._get_latest_successful_dataset_version(
            channel_dataset_ids=[channel_dataset.id]
        )
        version = latest_completed_version[dataset_db.id].last_completed_version

        if not version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="No completed versions found."
            )

        handler = await self._get_handler(dataset_db.source_id)
        config = handler.parse_data_set_config(dataset_db.details)

        config_changes = self._get_config_changes(version, config.indexing_hash)

        structure_hash, meta = await handler.get_structure_hash_and_metadata(
            dataset_config=dataset_db.details, auth_context=auth_context
        )
        if version.structure_hash == structure_hash:
            structure_change = None
        else:
            details = handler.get_structure_metadata_diff(version.structure_metadata, meta)
            structure_change = schemas.StructureChange(
                message="The dataset structure has changed.",
                last_version_hash=version.structure_hash,
                actual_hash=structure_hash,
                details=details,
            )

        if structure_change:
            data_changes = [
                schemas.DataChange(
                    message="The dataset structure has changed, so all data is considered changed.",
                    last_version_hash='N/A',
                    actual_hash='N/A',
                )
            ]
        else:
            dataset = await handler.get_dataset(
                entity_id=dataset_db.id_,
                title=dataset_db.title,
                config=dataset_db.details,
                auth_context=auth_context,
                allow_offline=False,
            )
            data_hashes = await self._get_data_hashes(
                dataset, auth_context=auth_context, allow_cached=False
            )
            data_changes = self._get_data_changes(version, data_hashes)

        return schemas.ChangesBetweenVersionAndActualData(
            config_changes=config_changes,
            data_changes=data_changes,
            structure_change=structure_change,
        )

    @staticmethod
    def _get_config_changes(
        version: schemas.ChannelDatasetVersion, config_hash: str
    ) -> list[schemas.ConfigChange]:
        if version.indexing_config_hash != config_hash:
            return [
                schemas.ConfigChange(
                    message="The dataset configuration (used during indexing) has changed.",
                    last_version_hash=version.indexing_config_hash,
                    actual_hash=config_hash,
                )
            ]
        return []

    @staticmethod
    def _get_data_changes(
        version: schemas.ChannelDatasetVersion, data_hashes: _DataHashes
    ) -> list[schemas.DataChange]:
        iterable = [
            ('Indicator', version.indicator_dimensions_hash, data_hashes.indicator_dimensions_hash),
            ('Special', version.special_dimensions_hash, data_hashes.special_dimensions_hash),
            (
                'Non-indicator',
                version.non_indicator_dimensions_hash,
                data_hashes.non_indicator_dimensions_hash,
            ),
        ]
        return [
            schemas.DataChange(
                message=f"Available data for the {name} dimensions has changed.",
                last_version_hash=old_hash,
                actual_hash=new_hash,
            )
            for name, old_hash, new_hash in iterable
            if old_hash != new_hash
        ]

    @staticmethod
    def _select_versions_to_clear(
        versions: list[models.ChannelDatasetVersion],
    ) -> list[int]:
        """Select old versions eligible for data clearing.

        Keeps the last 2 completed versions and returns IDs of the rest
        (completed beyond the last 2, plus any other final-status versions).
        """
        skip_last_completed = 2
        versions_to_clear: list[int] = []
        for version in versions:
            _status = version.preprocessing_status
            if _status == StatusEnum.COMPLETED:
                if skip_last_completed:
                    skip_last_completed -= 1
                else:
                    versions_to_clear.append(version.id)
            elif _status in StatusEnum.final_statuses():
                versions_to_clear.append(version.id)
        return versions_to_clear

    async def clear_channel_dataset_versions_data(self, channel_id: int, dataset_id: int):
        channel: schemas.Channel = await ChannelService(self._session).get_schema_by_id(channel_id)
        dataset: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel_id, dataset_id=dataset.id
        )

        # We don't need to clear older versions since most likely they have been already cleared
        versions = await self.get_channel_dataset_version_models(
            limit=20, offset=0, channel_dataset_id=channel_dataset.id
        )

        versions_to_clear = self._select_versions_to_clear(versions)

        if versions_to_clear:
            await self._clear_channel_dataset_data(
                channel, dataset_id=None, version_ids=versions_to_clear
            )
        else:
            _log.info("No versions to clear data for.")

    async def set_failed_status_for_channel_dataset_version(self) -> None:
        """Sets the status of all not-completed channel dataset versions to FAILED."""
        await set_failed_status(
            self._session,
            models.ChannelDatasetVersion,
            models.ChannelDatasetVersion.preprocessing_status,
            "preprocessing_status",
        )
        await self._session.commit()

    async def set_failed_status_for_stuck_jobs(self) -> None:
        """Sets the status of all stuck Job records to FAILED."""
        await set_failed_status(
            self._session,
            models.Job,
            models.Job.status,
            "status",
        )
        await self._session.commit()

    async def set_failed_status_for_stuck_auto_update_jobs(self) -> None:
        """Sets the status of all stuck AutoUpdateJob records to FAILED."""
        await set_failed_status(
            self._session,
            models.AutoUpdateJob,
            models.AutoUpdateJob.status,
            "status",
        )
        await self._session.commit()

    async def reload_all_indicators(
        self,
        background_tasks: BackgroundTasks,
        channel_id: int,
        auth_context: AuthContext,
        max_n_embeddings: int | None = None,
    ) -> list[schemas.ChannelDatasetExpanded]:
        channel: schemas.Channel = await ChannelService(self._session).get_schema_by_id(channel_id)
        channel_datasets: list[models.ChannelDataset] = await self.get_channel_dataset_models(
            limit=None, offset=0, channel_id=channel.id
        )
        dataset_ids = [ch_ds.dataset_id for ch_ds in channel_datasets]
        datasets: list[schemas.DataSet] = await self.get_datasets_schemas(
            limit=None, offset=0, auth_context=auth_context, ids=dataset_ids, allow_offline=False
        )
        harmonization_supported = self._is_harmonization_supported(channel)

        status_on_completion = (
            StatusEnum.QUEUED if harmonization_supported else StatusEnum.COMPLETED
        )
        new_versions: dict[int, models.ChannelDatasetVersion] = {}
        for ch_ds in channel_datasets:
            version = await self._create_new_channel_dataset_version(
                channel_dataset_id=ch_ds.id,
                reason="Manually initiated reindexing of all datasets in a channel.",
                preprocessing_status=StatusEnum.NOT_STARTED,
            )
            new_versions[version.channel_dataset_id] = version

        version_ids: set[int] = {version.id for version in new_versions.values()}
        for version in new_versions.values():
            background_tasks.add_task(
                reload_indicators_in_background_task,
                channel_dataset_version_id=version.id,
                version_ids=version_ids,
                reindex_indicators=True,
                harmonize_indicator=False,
                reindex_dimensions=True,
                auth_context=auth_context,
                max_n_embeddings=max_n_embeddings,
                status_on_completion=status_on_completion,
            )
            await self._update_channel_dataset_version_status(
                version, StatusEnum.QUEUED, do_commit=False
            )

        if harmonization_supported:
            for version in new_versions.values():
                background_tasks.add_task(
                    reload_indicators_in_background_task,
                    channel_dataset_version_id=version.id,
                    version_ids=version_ids,
                    reindex_indicators=True,
                    harmonize_indicator=True,
                    reindex_dimensions=False,
                    auth_context=auth_context,
                    max_n_embeddings=max_n_embeddings,
                )

        for ch_ds in channel_datasets:
            background_tasks.add_task(
                clear_channel_dataset_data_in_background_task,
                channel_dataset_id=ch_ds.id,
            )
            await self._update_channel_dataset_status(
                ch_ds, StatusEnum.NOT_STARTED, do_commit=False
            )

        await self._session.commit()
        for ch_ds in channel_datasets:
            await self._session.refresh(ch_ds)
        for version in new_versions.values():
            await self._session.refresh(version)

        last_completed_versions = await self._get_latest_successful_dataset_version(
            channel_dataset_ids=[ch_ds.id for ch_ds in channel_datasets]
        )
        latest_auto_update_jobs = await self._get_latest_auto_update_jobs(
            channel_dataset_ids=[ch_ds.id for ch_ds in channel_datasets]
        )
        return [
            ChannelDataSetSerializer.db_to_schema(
                item_db=ch_ds,
                dataset=next(ds for ds in datasets if ds.id == ch_ds.dataset_id),
                latest_version=schemas.ChannelDatasetVersion.model_validate(
                    new_versions[ch_ds.id], from_attributes=True
                ),
                last_completed_versions=last_completed_versions[ch_ds.dataset_id],
                last_auto_update_job=latest_auto_update_jobs.get(ch_ds.id),
            )
            for ch_ds in channel_datasets
        ]

    async def reload_indicators(
        self,
        background_tasks: BackgroundTasks,
        channel_id: int,
        dataset_id: int,
        auth_context: AuthContext,
        max_n_embeddings: int | None = None,
    ) -> schemas.ChannelDatasetExpanded:
        channel: schemas.Channel = await ChannelService(self._session).get_schema_by_id(channel_id)
        dataset: schemas.DataSet = await self.get_schema_by_id(dataset_id, auth_context)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel_id, dataset_id=dataset_id
        )

        version = await self._create_new_channel_dataset_version(
            channel_dataset_id=channel_dataset.id,
            reason="Manually initiated reindexing of a dataset in a channel.",
            preprocessing_status=StatusEnum.NOT_STARTED,
        )

        harmonization_supported = self._is_harmonization_supported(channel)

        background_tasks.add_task(
            reload_indicators_in_background_task,
            channel_dataset_version_id=version.id,
            version_ids=None,
            reindex_indicators=True,
            harmonize_indicator=False,
            reindex_dimensions=True,
            auth_context=auth_context,
            max_n_embeddings=max_n_embeddings,
            status_on_completion=(
                StatusEnum.QUEUED if harmonization_supported else StatusEnum.COMPLETED
            ),
        )
        await self._update_channel_dataset_version_status(version, StatusEnum.QUEUED)

        if harmonization_supported:
            background_tasks.add_task(
                reload_indicators_in_background_task,
                channel_dataset_version_id=version.id,
                version_ids=None,
                reindex_indicators=True,
                harmonize_indicator=True,
                reindex_dimensions=False,
                auth_context=auth_context,
                max_n_embeddings=max_n_embeddings,
            )

        background_tasks.add_task(
            clear_channel_dataset_data_in_background_task,
            channel_dataset_id=channel_dataset.id,
        )
        await self._update_channel_dataset_status(channel_dataset, StatusEnum.NOT_STARTED)

        latest_version = schemas.ChannelDatasetVersion.model_validate(version, from_attributes=True)
        last_completed_versions_mapping = (
            await self._get_latest_successful_channel_dataset_versions(
                channel_dataset_ids=[channel_dataset.id]
            )
        )
        last_completed_versions = last_completed_versions_mapping[channel_dataset.id]
        latest_auto_update_jobs = await self._get_latest_auto_update_jobs(
            channel_dataset_ids=[channel_dataset.id]
        )
        return ChannelDataSetSerializer.db_to_schema(
            channel_dataset,
            dataset,
            latest_version,
            last_completed_versions,
            latest_auto_update_jobs.get(channel_dataset.id),
        )

    @classmethod
    def _is_harmonization_supported(cls, channel: schemas.Channel) -> bool:
        return ChannelService.is_channel_hybrid(channel)

    @staticmethod
    async def _get_indicators_hash(
        dataset: base.DataSet, auth_context: AuthContext, allow_cached: bool
    ) -> str:
        indicators = await dataset.get_indicators(
            auth_context=auth_context, allow_cached=allow_cached
        )
        indicators_values = sorted(f"{i.query_id} {i.name}" for i in indicators)
        hash_value = await crc32_hash_incremental_async(indicators_values)
        return str(hash_value)

    @staticmethod
    async def _get_non_indicators_hash(dataset: base.DataSet) -> str:
        dimensions: Generator[base.CategoricalDimension] = (
            dim
            for dim in dataset.non_indicator_dimensions()
            if isinstance(dim, base.CategoricalDimension)
        )
        dimensions_values = sorted(
            f"{category_value.query_id} {category_value.name}"
            for dim in dimensions
            for category_value in dim.available_values
        )
        hash_value = await crc32_hash_incremental_async(dimensions_values)
        return str(hash_value)

    @staticmethod
    async def _get_special_dimensions_hash(dataset: base.DataSet) -> str | None:
        if not dataset.special_dimensions():
            return None
        dimensions: Generator[base.CategoricalDimension] = (
            dim
            for dim in dataset.special_dimensions().values()
            if isinstance(dim, base.CategoricalDimension)
        )
        dimensions_values = sorted(
            f"{category_value.query_id} {category_value.name}"
            for dim in dimensions
            for category_value in dim.available_values
        )
        hash_value = await crc32_hash_incremental_async(dimensions_values)
        return str(hash_value)

    async def _get_data_hashes(
        self, dataset: base.DataSet, auth_context: AuthContext, allow_cached: bool
    ) -> _DataHashes:
        indicator_dimensions_hash = await self._get_indicators_hash(
            dataset, auth_context=auth_context, allow_cached=allow_cached
        )
        non_indicator_dimensions_hash = await self._get_non_indicators_hash(dataset)
        special_dimensions_hash = await self._get_special_dimensions_hash(dataset)
        return _DataHashes(
            indicator_dimensions_hash=indicator_dimensions_hash,
            non_indicator_dimensions_hash=non_indicator_dimensions_hash,
            special_dimensions_hash=special_dimensions_hash,
        )

    @staticmethod
    async def _run_semantic_indexer(
        dataset: base.DataSet,
        source_id: int,
        vector_store: VectorStore,
        version_id: int,
        max_n_embeddings: int | None,
        auth_context: AuthContext,
    ) -> None:
        indicators = await dataset.get_indicators(auth_context=auth_context, allow_cached=True)
        _log.info(f"Loaded {len(indicators)} indicators.")
        if max_n_embeddings:
            indicators = indicators[:max_n_embeddings]  # for debug

        documents = (
            i.to_document({IndicatorDocumentMetadataFields.DATA_SOURCE_ID: source_id})
            for i in indicators
        )

        await vector_store.add_documents(documents, dataset_id=dataset.id, version_id=version_id)

    @staticmethod
    async def _run_hybrid_indexer(
        channel: schemas.Channel,
        vector_store: VectorStore,
        dataset: base.DataSet,
        version_id: int,
        version_ids: set[int],
        harmonize_indicator: bool,
        max_n_embeddings: int | None,
        auth_context: AuthContext,
    ) -> dict:
        matching_index = await ElasticSearchFactory.get_index(
            channel.matching_index_name, allow_creation=True
        )
        indicators_index = await ElasticSearchFactory.get_index(
            channel.indicators_index_name, allow_creation=True
        )
        if (data_query_config := channel.details.data_query) is None:
            raise ValueError(f"No data query configured for the channel {channel}")
        config = data_query_config.details.hybrid_search_config or HybridSearchConfig()

        indexer = Indexer(
            config,
            auth_context.api_key,
            matching_index,
            indicators_index,
            vector_store,
            normalize=not harmonize_indicator,
            harmonize=harmonize_indicator,
        )
        return await indexer.index(
            dataset,
            version_id=version_id,
            version_ids=version_ids,
            max_n_indicators=max_n_embeddings,
            auth_context=auth_context,
        )

    @staticmethod
    async def _index_non_indicator_dimensions(
        version_id: int,
        channel: schemas.Channel,
        dataset: base.DataSet,
        source_id: int,
        vector_store_factory: VectorStoreFactory,
        max_n_embeddings: int | None,
        auth_context: AuthContext,
    ) -> None:
        vector_store = await vector_store_factory.get_vector_store(
            collection_name=channel.non_indicator_dimensions_table_name,
            embedding_model_name=channel.llm_model,
            auth_context=auth_context,
        )

        dimensions = dataset.non_indicator_dimensions()
        documents = []
        for dimension in dimensions:
            if not isinstance(dimension, base.CategoricalDimension):
                continue
            category_values = dimension.available_values
            if max_n_embeddings is not None:
                category_values = category_values[:max_n_embeddings]

            for value in category_values:
                document = value.to_document()
                field_name = DimensionValueDocumentMetadataFields.DATA_SOURCE_ID
                document.metadata[field_name] = source_id
                documents.append(document)
        await vector_store.add_documents(documents, dataset_id=dataset.id, version_id=version_id)

        # ~~~~~ Special dimensions ~~~~~

        vector_store = await vector_store_factory.get_vector_store(
            collection_name=channel.special_dimensions_table_name,
            embedding_model_name=channel.llm_model,
            auth_context=auth_context,
        )

        documents = []
        for processor_id, dimension in dataset.special_dimensions().items():
            if not isinstance(dimension, base.CategoricalDimension):
                continue
            category_values = dimension.available_values
            if max_n_embeddings is not None:
                category_values = category_values[:max_n_embeddings]

            for value in category_values:
                document = value.to_document()
                field_name = SpecialDimensionValueDocumentMetadataFields.DATA_SOURCE_ID
                document.metadata[field_name] = source_id
                field_name = SpecialDimensionValueDocumentMetadataFields.PROCESSOR_ID
                document.metadata[field_name] = processor_id
                documents.append(document)
        await vector_store.add_documents(documents, dataset_id=dataset.id, version_id=version_id)

    async def _index_channel_indicators(
        self,
        channel: schemas.Channel,
        source_id: int,
        version_id: int,
        channel_dataset_id: int,
        version_ids: set[int] | None,
        harmonize_indicator: bool,
        max_n_embeddings: int | None,
        vector_store_factory: VectorStoreFactory,
        dataset: base.DataSet,
        auth_context: AuthContext,
    ) -> dict | None:
        vector_store = await vector_store_factory.get_vector_store(
            collection_name=channel.indicator_table_name,
            embedding_model_name=channel.llm_model,
            auth_context=auth_context,
        )

        if channel.details.data_query is None:
            _log.info(f"No data query found for version_id={version_id}, skipping indexing")
            return None

        indexer_version = channel.details.data_query.details.indexer_version
        _log.info(f"Indexer version: {indexer_version}")
        if indexer_version == schemas.IndexerVersion.hybrid:
            if version_ids is None:
                async with self._scoped_session():
                    res = await self.get_latest_successful_dataset_versions_for_channel(channel.id)
                    version_ids = {
                        v.last_completed_version.version_data_id
                        for v in res.values()
                        if v.last_completed_version is not None
                        and v.last_completed_version.channel_dataset_id != channel_dataset_id
                    }

            return await self._run_hybrid_indexer(
                channel=channel,
                vector_store=vector_store,
                dataset=dataset,
                version_id=version_id,
                version_ids=version_ids,
                harmonize_indicator=harmonize_indicator,
                max_n_embeddings=max_n_embeddings,
                auth_context=auth_context,
            )
        elif indexer_version == schemas.IndexerVersion.semantic:
            await self._run_semantic_indexer(
                dataset, source_id, vector_store, version_id, max_n_embeddings, auth_context
            )
            return None
        else:
            raise RuntimeError(f"Unknown indexer version: {indexer_version}")

    async def _invalid_version_status(self, version: models.ChannelDatasetVersion) -> bool:
        """Check if the version is in a valid state to start processing."""

        if version.preprocessing_status == StatusEnum.FAILED:
            # This can happen if previous job failed (e.g. reindexing indicators before harmonization)
            _log.warning(f"Version {version} is in FAILED state. Skipping reindexing.")
            return True
        elif version.preprocessing_status != StatusEnum.QUEUED:
            # If the previous job failed and the status could not be updated to FAILED
            _log.warning(f"Version {version} is not in QUEUED state. Skipping reindexing.")
            reason_for_failure = (
                version.reason_for_failure  # Use existing reason if available
                or f"Cannot start processing a version that is in {version.preprocessing_status} state."
            )
            await self._update_channel_dataset_version_status(
                version, StatusEnum.FAILED, reason_for_failure=reason_for_failure
            )
            return True

        return False

    async def reload_channel_dataset_in_background(
        self,
        channel_dataset_version_id: int,
        version_ids: set[int] | None,
        reindex_indicators: bool,
        harmonize_indicator: bool,
        reindex_dimensions: bool,
        auth_context: AuthContext,
        max_n_embeddings: int | None,
        status_on_completion: StatusEnum = StatusEnum.COMPLETED,
    ) -> None:
        try:
            # Phase A: DB reads — load entities, convert to schemas/session-independent objects
            async with self._scoped_session():
                version = await self._get_channel_dataset_version_or_raise(
                    channel_dataset_version_id
                )
                _log.info(f"Start processing {version}")

                if await self._invalid_version_status(version):
                    return

                await self._update_channel_dataset_version_status(
                    version, new_status=StatusEnum.IN_PROGRESS
                )

                channel_dataset = await self._get_channel_dataset_model_or_raise(
                    version.channel_dataset_id
                )
                channel_dataset_id = channel_dataset.id
                channel_id = channel_dataset.channel_id
                dataset_id = channel_dataset.dataset_id
                _log.info(
                    f"Processing version(id={version.id}, version={version.version})"
                    f" of {channel_dataset}"
                )

                db_dataset: models.DataSet = await self.get_model_by_id(dataset_id)
                channel = await ChannelService(self._session).get_schema_by_id(channel_id)
                dataset_config = version.resolved_config or db_dataset.details
                entity_id = db_dataset.id_
                source_id = db_dataset.source_id
                dataset_title = db_dataset.title

                handler = await self._get_handler(db_dataset.source_id)

            # Phase B: Network I/O — no session needed
            dataset = await handler.get_dataset(
                entity_id=entity_id,
                title=dataset_title,
                config=dataset_config,
                auth_context=auth_context,
                allow_offline=False,
            )

            resolved_config = dataset.get_resolved_config()

            hashes_to_store: tuple[str, str, dict, _DataHashes] | None = None
            if reindex_dimensions or (reindex_indicators and not harmonize_indicator):
                config_hash = dataset.config.indexing_hash
                structure_hash, meta = await handler.get_structure_hash_and_metadata(
                    dataset_config=dataset_config, auth_context=auth_context
                )
                data_hashes = await self._get_data_hashes(dataset, auth_context, allow_cached=True)
                hashes_to_store = (config_hash, structure_hash, meta, data_hashes)

            # Phase C: DB writes — store resolved config, hashes, metadata
            async with self._scoped_session():
                version = await self._get_channel_dataset_version_or_raise(
                    channel_dataset_version_id
                )
                await self._set_resolved_config(version, resolved_config)

                if hashes_to_store is not None:
                    await self._set_version_hashes_and_metadata(version, *hashes_to_store)

            # Phase D: Vector indexing — no DB session needed, vector stores manage their own connections
            vector_store_factory = VectorStoreFactory()
            indexing_stats: dict | None = None

            if reindex_dimensions:
                await self._index_non_indicator_dimensions(
                    version_id=channel_dataset_version_id,
                    channel=channel,
                    dataset=dataset,
                    source_id=source_id,
                    vector_store_factory=vector_store_factory,
                    max_n_embeddings=max_n_embeddings,
                    auth_context=auth_context,
                )

            if reindex_indicators:
                indexing_stats = await self._index_channel_indicators(
                    channel=channel,
                    source_id=source_id,
                    version_id=channel_dataset_version_id,
                    channel_dataset_id=channel_dataset_id,
                    version_ids=version_ids,
                    harmonize_indicator=harmonize_indicator,
                    max_n_embeddings=max_n_embeddings,
                    vector_store_factory=vector_store_factory,
                    dataset=dataset,
                    auth_context=auth_context,
                )

            # Phase E: Persist indexing stats and update status
            async with self._scoped_session():
                version = await self._get_channel_dataset_version_or_raise(
                    channel_dataset_version_id
                )
                if indexing_stats:
                    version.indexing_stats = {**(version.indexing_stats or {}), **indexing_stats}
                await self._update_channel_dataset_version_status(
                    version, new_status=status_on_completion
                )

                if status_on_completion is StatusEnum.COMPLETED:
                    channel_dataset = await self._get_channel_dataset_model_or_raise(
                        channel_dataset_id
                    )
                    await self._update_channel_dataset_status(
                        channel_dataset, new_status=StatusEnum.QUEUED
                    )
                _log.info(
                    f'Finished processing version_id={channel_dataset_version_id}'
                    f' of channel_dataset_id={channel_dataset_id}'
                )
        except Exception as e:
            _log.exception(f"Failed to reindex version_id={channel_dataset_version_id}")
            async with self._scoped_session():
                version = await self._get_channel_dataset_version_or_raise(
                    channel_dataset_version_id
                )
                await self._update_channel_dataset_version_status(
                    version,
                    new_status=StatusEnum.FAILED,
                    reason_for_failure=format_exception_reason(e),
                )

    async def clear_channel_dataset_data_in_background(self, channel_dataset_id: int) -> None:
        _log.info(f"Clear data after reindexing channel_dataset_id={channel_dataset_id}")
        try:
            # Phase A: DB reads — load entities, set status, determine versions to clear
            async with self._scoped_session():
                channel_dataset = await self._get_channel_dataset_model_or_raise(channel_dataset_id)
                await self._update_channel_dataset_status(
                    channel_dataset, new_status=StatusEnum.IN_PROGRESS
                )

                channel: schemas.Channel = await ChannelService(self._session).get_schema_by_id(
                    channel_dataset.channel_id
                )

                versions = await self.get_channel_dataset_version_models(
                    limit=20, offset=0, channel_dataset_id=channel_dataset.id
                )
                versions_to_clear = self._select_versions_to_clear(versions)

            # Phase B: Vector store + elastic clearing — no DB session held
            if versions_to_clear:
                await self._clear_channel_dataset_data(
                    channel, dataset_id=None, version_ids=versions_to_clear
                )
            else:
                _log.info("No versions to clear data for.")

            # Phase C: DB write — update status to COMPLETED
            async with self._scoped_session():
                channel_dataset = await self._get_channel_dataset_model_or_raise(channel_dataset_id)
                await self._update_channel_dataset_status(
                    channel_dataset, new_status=StatusEnum.COMPLETED
                )
        except Exception:
            _log.exception(
                f"Failed to clear data after reindexing channel_dataset_id={channel_dataset_id}"
            )
            async with self._scoped_session():
                channel_dataset = await self._get_channel_dataset_model_or_raise(channel_dataset_id)
                await self._update_channel_dataset_status(
                    channel_dataset, new_status=StatusEnum.FAILED
                )

    async def _get_deduplication_status_by_versions(
        self,
        non_indicator_dims_store: EmbeddinglessVectorStore,
        special_dims_store: EmbeddinglessVectorStore,
        indicator_dims_store: EmbeddinglessVectorStore,
        versions: set[int],
    ) -> schemas.DeduplicationStatus:
        non_indicator_has_duplicates, non_indicator_count = (
            await non_indicator_dims_store.has_duplicates_in_versions(version_ids=versions)
        )
        special_has_duplicates, special_count = await special_dims_store.has_duplicates_in_versions(
            version_ids=versions
        )
        _, indicator_count = await indicator_dims_store.has_duplicates_in_versions(
            version_ids=versions
        )

        # we only consider non-indicator and special dimensions for deduplication requirement
        deduplication_required = non_indicator_has_duplicates or special_has_duplicates
        total_duplicates = non_indicator_count + special_count + indicator_count

        return schemas.DeduplicationStatus(
            deduplication_required=deduplication_required,
            total_duplicate_count=total_duplicates,
            non_indicator_dimensions_duplicate_count=non_indicator_count,
            special_dimensions_duplicate_count=special_count,
            indicator_dimensions_duplicate_count=indicator_count,
        )

    async def _get_full_deduplication_status(
        self,
        non_indicator_dims_store: EmbeddinglessVectorStore,
        special_dims_store: EmbeddinglessVectorStore,
        indicator_dims_store: EmbeddinglessVectorStore,
    ) -> schemas.DeduplicationStatus:
        non_indicator_has_duplicates, non_indicator_count = (
            await non_indicator_dims_store.has_duplicates()
        )
        special_has_duplicates, special_count = await special_dims_store.has_duplicates()
        _, indicator_count = await indicator_dims_store.has_duplicates()

        # we only consider non-indicator and special dimensions for deduplication requirement
        deduplication_required = non_indicator_has_duplicates or special_has_duplicates
        total_duplicates = non_indicator_count + special_count + indicator_count

        return schemas.DeduplicationStatus(
            deduplication_required=deduplication_required,
            total_duplicate_count=total_duplicates,
            non_indicator_dimensions_duplicate_count=non_indicator_count,
            special_dimensions_duplicate_count=special_count,
            indicator_dimensions_duplicate_count=indicator_count,
        )

    async def _check_latest_versions_status(
        self,
        channel: models.Channel,
        non_indicator_dims_store: EmbeddinglessVectorStore,
        special_dims_store: EmbeddinglessVectorStore,
        indicator_dims_store: EmbeddinglessVectorStore,
    ) -> schemas.ChannelIndexStatus:
        latest_successful_versions = await self.get_latest_successful_dataset_versions_for_channel(
            channel_id=channel.id
        )
        versions = {
            v.last_completed_version.version_data_id
            for v in latest_successful_versions.values()
            if v.last_completed_version is not None
        }

        deduplication_status = await self._get_deduplication_status_by_versions(
            non_indicator_dims_store,
            special_dims_store,
            indicator_dims_store,
            versions,
        )
        sizes = schemas.VectorStoreSizes(
            non_indicator_dimensions_size=await non_indicator_dims_store.get_size(
                version_ids=versions
            ),
            special_dimensions_size=await special_dims_store.get_size(version_ids=versions),
            indicator_dimensions_size=await indicator_dims_store.get_size(version_ids=versions),
        )

        vector_store_status = schemas.VectorStoreStatus(
            deduplication=deduplication_status,
            sizes=sizes,
        )
        return schemas.ChannelIndexStatus(
            vector_store=vector_store_status,
            scope=ChannelIndexStatusScope.LATEST_COMPLETED_VERSIONS,
        )

    async def _check_full_index_status(
        self,
        channel: models.Channel,
        non_indicator_dims_store: EmbeddinglessVectorStore,
        special_dims_store: EmbeddinglessVectorStore,
        indicator_dims_store: EmbeddinglessVectorStore,
    ) -> schemas.ChannelIndexStatus:
        deduplication_status = await self._get_full_deduplication_status(
            non_indicator_dims_store,
            special_dims_store,
            indicator_dims_store,
        )
        sizes = schemas.VectorStoreSizes(
            non_indicator_dimensions_size=await non_indicator_dims_store.get_total_size(),
            special_dimensions_size=await special_dims_store.get_total_size(),
            indicator_dimensions_size=await indicator_dims_store.get_total_size(),
        )

        vector_store_status = schemas.VectorStoreStatus(
            deduplication=deduplication_status,
            sizes=sizes,
        )
        return schemas.ChannelIndexStatus(
            vector_store=vector_store_status, scope=ChannelIndexStatusScope.FULL
        )

    async def check_index_status(
        self,
        channel_id: int,
        scope: schemas.ChannelIndexStatusScope,
    ) -> schemas.ChannelIndexStatus:
        """Checks index status for channel"""
        channel = await ChannelService(self._session).get_model_by_id(channel_id)
        vector_store_factory = VectorStoreFactory()

        non_indicator_dims_store = await vector_store_factory.get_embeddingless_vector_store(
            collection_name=channel.non_indicator_dimensions_table_name,
        )
        special_dims_store = await vector_store_factory.get_embeddingless_vector_store(
            collection_name=channel.special_dimensions_table_name,
        )
        indicator_dims_store = await vector_store_factory.get_embeddingless_vector_store(
            collection_name=channel.indicator_table_name,
        )

        if scope == schemas.ChannelIndexStatusScope.FULL:
            return await self._check_full_index_status(
                channel=channel,
                non_indicator_dims_store=non_indicator_dims_store,
                special_dims_store=special_dims_store,
                indicator_dims_store=indicator_dims_store,
            )
        elif scope == schemas.ChannelIndexStatusScope.LATEST_COMPLETED_VERSIONS:
            return await self._check_latest_versions_status(
                channel=channel,
                non_indicator_dims_store=non_indicator_dims_store,
                special_dims_store=special_dims_store,
                indicator_dims_store=indicator_dims_store,
            )
        else:
            raise ValueError(f"Unknown scope: {scope}")

    async def trigger_auto_update(
        self,
        background_tasks: BackgroundTasks,
        channel_id: int,
        dataset_id: int,
        auth_context: AuthContext,
    ) -> schemas.AutoUpdateJob:
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel_id, dataset_id=dataset_id
        )

        job = models.AutoUpdateJob(
            channel_dataset_id=channel_dataset.id,
            status=StatusEnum.QUEUED,
        )
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)

        background_tasks.add_task(
            auto_update_in_background_task,
            auto_update_job_id=job.id,
            auth_context=auth_context,
        )

        return schemas.AutoUpdateJob.model_validate(job, from_attributes=True)

    async def get_auto_update_job_by_id(self, job_id: int) -> schemas.AutoUpdateJob:
        job: models.AutoUpdateJob | None = await self._session.get(models.AutoUpdateJob, job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Auto-update job not found"
            )
        return schemas.AutoUpdateJob.model_validate(job, from_attributes=True)

    async def get_auto_update_jobs(
        self,
        channel_dataset_id: int,
        limit: int,
        offset: int,
    ) -> list[schemas.AutoUpdateJob]:
        """Get paginated list of auto-update jobs for a channel dataset."""
        query = (
            select(models.AutoUpdateJob)
            .where(models.AutoUpdateJob.channel_dataset_id == channel_dataset_id)
            .order_by(models.AutoUpdateJob.id.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.scalars(query)
        return [
            schemas.AutoUpdateJob.model_validate(job, from_attributes=True) for job in result.all()
        ]

    async def get_auto_update_jobs_count(self, channel_dataset_id: int) -> int:
        """Get total count of auto-update jobs for a channel dataset."""
        query = (
            select(func.count())
            .select_from(models.AutoUpdateJob)
            .where(models.AutoUpdateJob.channel_dataset_id == channel_dataset_id)
        )
        return await self._session.scalar(query) or 0

    async def create_auto_update_jobs(self, channel_ids: list[int]) -> list[schemas.AutoUpdateJob]:
        """Create AutoUpdateJob records for all datasets in the given channels.

        Returns a list of created job schemas.
        """
        result = await self._session.execute(
            select(models.ChannelDataset)
            .where(models.ChannelDataset.channel_id.in_(channel_ids))
            .options(selectinload(models.ChannelDataset.channel))
        )
        channel_datasets = list(result.scalars().all())

        jobs: list[models.AutoUpdateJob] = []
        for cd in channel_datasets:
            job = models.AutoUpdateJob(channel_dataset_id=cd.id, status=StatusEnum.QUEUED)
            self._session.add(job)
            jobs.append(job)
        await self._session.commit()

        # Log per-channel summary
        channels_by_id = {cd.channel.id: cd.channel for cd in channel_datasets}
        counts = Counter(cd.channel_id for cd in channel_datasets)
        for ch_id, count in counts.items():
            ch = channels_by_id[ch_id]
            _log.info(
                f"Created {count} auto-update job(s) for channel '{ch.deployment_id}' (id={ch_id})"
            )

        return [schemas.AutoUpdateJob.model_validate(job, from_attributes=True) for job in jobs]

    async def get_reindex_channel_ids(self, job_ids: list[int]) -> set[int]:
        """Get channel IDs that had at least one REINDEX_TRIGGERED result."""
        result = await self._session.execute(
            select(models.AutoUpdateJob)
            .where(models.AutoUpdateJob.id.in_(job_ids))
            .where(models.AutoUpdateJob.result == AutoUpdateResult.REINDEX_TRIGGERED)
            .options(selectinload(models.AutoUpdateJob.channel_dataset))
        )
        jobs = list(result.scalars().all())
        return {j.channel_dataset.channel_id for j in jobs}

    async def get_auto_update_results(self, job_ids: list[int]) -> list[AutoUpdateChannelResult]:
        """Collect per-channel auto-update results."""
        result = await self._session.execute(
            select(models.AutoUpdateJob)
            .where(models.AutoUpdateJob.id.in_(job_ids))
            .options(
                selectinload(models.AutoUpdateJob.channel_dataset).selectinload(
                    models.ChannelDataset.channel
                ),
                selectinload(models.AutoUpdateJob.created_version),
            )
        )
        jobs = list(result.scalars().all())

        channel_jobs: defaultdict[int, list[models.AutoUpdateJob]] = defaultdict(list)
        for job in jobs:
            channel_jobs[job.channel_dataset.channel.id].append(job)

        results: list[AutoUpdateChannelResult] = []
        for channel_id, ch_jobs in channel_jobs.items():
            ch = ch_jobs[0].channel_dataset.channel
            results.append(
                AutoUpdateChannelResult(
                    channel_id=channel_id,
                    deployment_id=ch.deployment_id,
                    total=len(ch_jobs),
                    failed=sum(1 for j in ch_jobs if j.status == StatusEnum.FAILED),
                    summary=self._format_result_summary(ch_jobs),
                    failed_reasons=[
                        f"job {j.id}: {j.reason_for_failure}"
                        for j in ch_jobs
                        if j.status == StatusEnum.FAILED
                    ],
                )
            )
        return results

    @staticmethod
    def _format_result_summary(jobs: list[models.AutoUpdateJob]) -> str:
        """Build a human-readable summary of auto-update results."""
        reindex_statuses: dict[str, Counter[str]] = defaultdict(Counter)
        job_statuses: list[str] = []
        for job in jobs:
            job_status = job.result.value if job.result else job.status.value
            job_statuses.append(job_status)
            if job.created_version is not None:
                reindex_statuses[job_status][job.created_version.preprocessing_status.value] += 1
        result_counts = Counter(job_statuses)

        parts: list[str] = []
        for job_status, count in result_counts.most_common():
            part = f"{count} {job_status}"
            if job_status in reindex_statuses:
                breakdown = ", ".join(f"{c} {s}" for s, c in reindex_statuses[job_status].items())
                part += f" ({breakdown})"
            parts.append(part)
        return ", ".join(parts)

    async def _get_auto_update_job_or_raise(self, auto_update_job_id: int) -> models.AutoUpdateJob:
        job: models.AutoUpdateJob | None = await self._session.get(
            models.AutoUpdateJob, auto_update_job_id
        )
        if job is None:
            raise RuntimeError(f"Auto-update job {auto_update_job_id} not found")
        return job

    async def _set_auto_update_job_status(
        self,
        job: models.AutoUpdateJob,
        status: StatusEnum,
        *,
        details: str | None = None,
        result: schemas.AutoUpdateResult | None = None,
        reason_for_failure: str | None = None,
    ) -> None:
        job.status = status

        if details is not None:
            job.details = details
        if reason_for_failure is not None:
            job.reason_for_failure = reason_for_failure
        if result is not None:
            job.result = result

        job.updated_at = func.now()
        await self._session.commit()
        await self._session.refresh(job)

    async def _get_last_completed_version(
        self, channel_dataset_id: int
    ) -> schemas.ChannelDatasetVersion | None:
        last_completed_mapping = await self._get_latest_successful_channel_dataset_versions(
            channel_dataset_ids=[channel_dataset_id]
        )
        last_completed_versions = last_completed_mapping.get(channel_dataset_id)
        return last_completed_versions.last_completed_version if last_completed_versions else None

    async def _get_data_changed(
        self,
        handler: base.DataSourceHandler,
        channel_dataset: models.ChannelDataset,
        new_resolved_config: dict,
        last_completed: schemas.ChannelDatasetVersion,
        auth_context: AuthContext,
    ) -> tuple[bool, str]:
        dataset = await handler.get_dataset(
            entity_id=channel_dataset.dataset.id_,
            title=channel_dataset.dataset.title,
            config=new_resolved_config,
            auth_context=auth_context,
            allow_offline=False,
        )
        data_hashes = await self._get_data_hashes(
            dataset, auth_context=auth_context, allow_cached=False
        )

        changed = []
        if last_completed.indicator_dimensions_hash != data_hashes.indicator_dimensions_hash:
            changed.append("indicator")
        if (
            last_completed.non_indicator_dimensions_hash
            != data_hashes.non_indicator_dimensions_hash
        ):
            changed.append("non-indicator")

        if last_completed.special_dimensions_hash != data_hashes.special_dimensions_hash:
            changed.append("special")

        if changed:
            changed_dimensions = ", ".join(changed)
            return True, f"Data in {changed_dimensions} dimensions has changed."
        else:
            return False, "Data has not changed."

    async def _create_config_only_version(
        self,
        channel_dataset: models.ChannelDataset,
        last_completed: schemas.ChannelDatasetVersion,
        new_resolved_config: dict,
    ) -> models.ChannelDatasetVersion:
        """Create a new version with updated config, reusing data from the last completed version.

        Used when the config changed (e.g., URN version updated) but the actual data is unchanged,
        so no reindexing is needed.
        """
        new_version = models.ChannelDatasetVersion(
            channel_dataset_id=channel_dataset.id,
            preprocessing_status=StatusEnum.COMPLETED,
            pointer_to=last_completed.version_data_id,
            creation_reason="Auto-update: config updated without reindexing",
            resolved_config=new_resolved_config,
            indexing_config_hash=last_completed.indexing_config_hash,
            structure_metadata=last_completed.structure_metadata,
            structure_hash=last_completed.structure_hash,
            indicator_dimensions_hash=last_completed.indicator_dimensions_hash,
            non_indicator_dimensions_hash=last_completed.non_indicator_dimensions_hash,
            special_dimensions_hash=last_completed.special_dimensions_hash,
        )
        self._session.add(new_version)
        await self._session.commit()
        await self._session.refresh(new_version)
        return new_version

    async def _prepare_auto_update_reindex(
        self,
        job: models.AutoUpdateJob,
        channel_dataset: models.ChannelDataset,
        new_resolved_config: dict,
        details: str,
    ) -> _ReindexParams:
        """Create a new version and prepare parameters for reindexing.

        Must be called inside an active ``_scoped_session()`` block.
        """
        new_version = await self._create_new_channel_dataset_version(
            channel_dataset_id=channel_dataset.id,
            reason="Auto-update triggered reindexing",
            preprocessing_status=StatusEnum.NOT_STARTED,
            resolved_config=new_resolved_config,
        )
        job.created_version_id = new_version.id
        await self._set_auto_update_job_status(
            job,
            StatusEnum.IN_PROGRESS,
            details=details,
            result=schemas.AutoUpdateResult.REINDEX_TRIGGERED,
        )

        channel = ChannelSerializer.db_to_schema(channel_dataset.channel)
        harmonization_supported = self._is_harmonization_supported(channel)
        status_on_completion = (
            StatusEnum.QUEUED if harmonization_supported else StatusEnum.COMPLETED
        )

        await self._update_channel_dataset_version_status(new_version, StatusEnum.QUEUED)

        return _ReindexParams(
            version_id=new_version.id,
            channel_dataset_id=channel_dataset.id,
            harmonization_supported=harmonization_supported,
            status_on_completion=status_on_completion,
        )

    async def _execute_auto_update_reindex(
        self,
        auto_update_job_id: int,
        params: _ReindexParams,
        auth_context: AuthContext,
    ) -> None:
        """Run reindexing background tasks. Must be called outside ``_scoped_session()``."""
        await self.reload_channel_dataset_in_background(
            channel_dataset_version_id=params.version_id,
            version_ids=None,
            reindex_indicators=True,
            harmonize_indicator=False,
            reindex_dimensions=True,
            auth_context=auth_context,
            max_n_embeddings=None,
            status_on_completion=params.status_on_completion,
        )

        if params.harmonization_supported:
            await self.reload_channel_dataset_in_background(
                channel_dataset_version_id=params.version_id,
                version_ids=None,
                reindex_indicators=True,
                harmonize_indicator=True,
                reindex_dimensions=False,
                auth_context=auth_context,
                max_n_embeddings=None,
            )

        await self.clear_channel_dataset_data_in_background(params.channel_dataset_id)

        async with self._scoped_session():
            job = await self._get_auto_update_job_or_raise(auto_update_job_id)
            await self._set_auto_update_job_status(
                job, StatusEnum.COMPLETED, result=schemas.AutoUpdateResult.REINDEX_TRIGGERED
            )

    async def _mark_auto_update_job_failed(self, auto_update_job_id: int, reason: str) -> None:
        async with self._scoped_session():
            job = await self._session.get(models.AutoUpdateJob, auto_update_job_id)
            if job is not None:
                job.status = StatusEnum.FAILED
                job.reason_for_failure = reason
                await self._session.commit()

    async def process_auto_update_job(
        self,
        auto_update_job_id: int,
        auth_context: AuthContext,
    ) -> None:
        """Process an auto-update job in the background."""
        _log.info(f"Processing auto-update job {auto_update_job_id}")
        try:
            # Phase A: DB reads — load job, channel_dataset, check state
            async with self._scoped_session():
                job: models.AutoUpdateJob | None = await self._session.get(
                    models.AutoUpdateJob, auto_update_job_id
                )
                if job is None:
                    _log.error(f"Auto-update job {auto_update_job_id} not found")
                    return

                await self._set_auto_update_job_status(job=job, status=StatusEnum.IN_PROGRESS)

                channel_dataset_id = job.channel_dataset_id
                channel_dataset = await self._get_channel_dataset_model_or_raise(channel_dataset_id)
                _log.info(
                    f"Auto-update job {auto_update_job_id}: channel_dataset_id={channel_dataset_id}"
                )

                if await self._is_indexing_in_progress(channel_dataset_id):
                    msg = f"Channel dataset {channel_dataset_id} is currently being indexed."
                    await self._set_auto_update_job_status(
                        job=job, status=StatusEnum.FAILED, reason_for_failure=msg
                    )
                    return

                last_completed = await self._get_last_completed_version(channel_dataset_id)
                if last_completed is None:
                    await self._set_auto_update_job_status(
                        job,
                        StatusEnum.COMPLETED,
                        result=schemas.AutoUpdateResult.NO_COMPLETED_VERSION,
                    )
                    return

                job.base_version_id = last_completed.id

                db_dataset: models.DataSet = await self.get_model_by_id(channel_dataset.dataset_id)
                handler = await self._get_handler(db_dataset.source_id)
                dataset_details = db_dataset.details

            # Phase B: Network I/O — no session needed
            # last_completed is a schema, so resolved_config is accessible without a session
            if last_completed.resolved_config is None:
                details, new_resolved_config = await handler.resolve_config(
                    config=dataset_details, auth_context=auth_context
                )
            else:
                details, new_resolved_config = await handler.reresolve_config(
                    config=dataset_details,
                    previous_resolved_config=last_completed.resolved_config,
                    auth_context=auth_context,
                )

            validation_result = await handler.validate_dataset_config(
                new_resolved_config, auth_context=auth_context, mode="return"
            )
            if not validation_result.is_valid:
                async with self._scoped_session():
                    job = await self._get_auto_update_job_or_raise(auto_update_job_id)
                    await self._set_auto_update_job_status(
                        job=job,
                        status=StatusEnum.COMPLETED,
                        result=schemas.AutoUpdateResult.CONFIG_INCOMPATIBLE,
                        details=details,
                        reason_for_failure="; ".join(validation_result.errors),
                    )
                return

            structure_hash, structure_meta = await handler.get_structure_hash_and_metadata(
                dataset_config=new_resolved_config, auth_context=auth_context
            )
            structure_changed = last_completed.structure_hash != structure_hash

            if structure_changed:
                structure_changes = handler.get_structure_metadata_diff(
                    old_metadata=last_completed.structure_metadata, new_metadata=structure_meta
                )
                details += f" Structure has changed: {structure_changes}."
                data_changed = None
            else:
                async with self._scoped_session():
                    channel_dataset = await self._get_channel_dataset_model_or_raise(
                        channel_dataset_id
                    )
                    await self._session.refresh(
                        channel_dataset, attribute_names=["channel", "dataset"]
                    )
                    data_changed, data_details = await self._get_data_changed(
                        handler, channel_dataset, new_resolved_config, last_completed, auth_context
                    )
                details += f" Structure has not changed. {data_details}"

            # Phase C: DB writes — prepare reindex or update status
            reindex_params: _ReindexParams | None = None
            async with self._scoped_session():
                job = await self._get_auto_update_job_or_raise(auto_update_job_id)
                channel_dataset = await self._get_channel_dataset_model_or_raise(channel_dataset_id)
                await self._session.refresh(channel_dataset, attribute_names=["channel", "dataset"])
                job.base_version_id = last_completed.id

                if structure_changed or data_changed:
                    reindex_params = await self._prepare_auto_update_reindex(
                        job, channel_dataset, new_resolved_config, details
                    )
                elif new_resolved_config != last_completed.resolved_config:
                    new_version = await self._create_config_only_version(
                        channel_dataset, last_completed, new_resolved_config
                    )
                    job.created_version_id = new_version.id
                    await self._set_auto_update_job_status(
                        job,
                        StatusEnum.COMPLETED,
                        details=details,
                        result=schemas.AutoUpdateResult.CONFIG_UPDATED,
                    )
                else:
                    await self._set_auto_update_job_status(
                        job,
                        StatusEnum.COMPLETED,
                        details=details,
                        result=schemas.AutoUpdateResult.NO_CHANGES,
                    )

            # Phase D: Trigger reindex tasks (outside session scope)
            if reindex_params is not None:
                await self._execute_auto_update_reindex(
                    auto_update_job_id=auto_update_job_id,
                    params=reindex_params,
                    auth_context=auth_context,
                )

        except asyncio.CancelledError:
            # A per-task timeout in the @background_task decorator cancels this
            # coroutine, surfacing here as CancelledError (a BaseException that the
            # `except Exception` branch below does NOT catch). Without this branch the
            # job would be left stuck in IN_PROGRESS. Mark it FAILED — shielded so the
            # write survives the cancellation — then re-raise so the decorator's
            # timeout machinery still runs.
            _log.error(f"Auto-update job {auto_update_job_id} was cancelled (likely timed out)")
            await asyncio.shield(
                self._mark_auto_update_job_failed(
                    auto_update_job_id, "Job cancelled (likely timed out)"
                )
            )
            raise
        except Exception as e:
            _log.exception(f"Failed to process auto-update job {auto_update_job_id}")
            await self._mark_auto_update_job_failed(auto_update_job_id, format_exception_reason(e))


@background_task
async def auto_update_in_background_task(
    auto_update_job_id: int,
    auth_context: AuthContext,
) -> None:
    """Background task wrapper for auto-update job processing."""
    try:
        service = AdminPortalDataSetService()
        await service.process_auto_update_job(
            auto_update_job_id=auto_update_job_id,
            auth_context=auth_context,
        )
    except Exception as e:
        _log.exception(e)


@background_task
async def reload_indicators_in_background_task(
    channel_dataset_version_id: int,
    version_ids: set[int] | None,
    reindex_indicators: bool,
    harmonize_indicator: bool,
    reindex_dimensions: bool,
    auth_context: AuthContext,
    max_n_embeddings: int | None,
    status_on_completion: StatusEnum = StatusEnum.COMPLETED,
) -> None:
    try:
        service = AdminPortalDataSetService()
        await service.reload_channel_dataset_in_background(
            channel_dataset_version_id=channel_dataset_version_id,
            version_ids=version_ids,
            reindex_indicators=reindex_indicators,
            harmonize_indicator=harmonize_indicator,
            reindex_dimensions=reindex_dimensions,
            auth_context=auth_context,
            max_n_embeddings=max_n_embeddings,
            status_on_completion=status_on_completion,
        )
    except Exception as e:
        _log.exception(e)


@background_task
async def clear_channel_dataset_data_in_background_task(channel_dataset_id: int) -> None:
    try:
        service = AdminPortalDataSetService()
        await service.clear_channel_dataset_data_in_background(channel_dataset_id)
    except Exception as e:
        _log.exception(e)

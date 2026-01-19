import json
import logging
import os.path
import uuid
import zipfile
from collections.abc import Generator, Iterable
from typing import Any, NamedTuple

import yaml
from fastapi import BackgroundTasks, HTTPException, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import func, text, update

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.admin.settings.exim import ExImSettings, JobsConfig
from statgpt.common import utils
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data import base
from statgpt.common.data.base.dataset import DataSetConfigType
from statgpt.common.hybrid_indexer import Indexer
from statgpt.common.schemas import ChannelIndexStatusScope, HybridSearchConfig
from statgpt.common.schemas import PreprocessingStatusEnum as StatusEnum
from statgpt.common.services import ChannelDataSetSerializer, DataSetSerializer, DataSetService
from statgpt.common.services.dataset import LastCompletedVersions
from statgpt.common.settings.document import (
    DimensionValueDocumentMetadataFields,
    IndicatorDocumentMetadataFields,
    SpecialDimensionValueDocumentMetadataFields,
)
from statgpt.common.utils import async_utils, crc32_hash_incremental_async
from statgpt.common.utils.elastic import ElasticIndex, ElasticSearchFactory, SearchResult
from statgpt.common.vectorstore import VectorStore, VectorStoreFactory

from .background_tasks import background_task
from .channel import AdminPortalChannelService as ChannelService
from .data_source import AdminPortalDataSourceService as DataSourceService
from .data_source import DataSourceTypeService

_log = logging.getLogger(__name__)


class _DataHashes(NamedTuple):
    indicator_dimensions_hash: str
    non_indicator_dimensions_hash: str
    special_dimensions_hash: str | None


class AdminPortalDataSetService(DataSetService):

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, None)  # No need for session lock in Admin Portal

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
        vector_store_factory = VectorStoreFactory(session=self._session)

        # collect last completed version ids
        version_ids: set[int] = set()
        for versions in latest_completed_versions.values():
            if versions.last_completed_version:
                version_ids.add(versions.last_completed_version.version_data_id)

        _log.info(f"Exporting {len(version_ids)} version(s): {sorted(version_ids)}")

        collections = [
            channel.available_dimensions_table_name,
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
                            for field in schemas.DataSetUpdate.model_fields.keys()
                            if getattr(parsed_dataset, field) != getattr(dataset, field)
                        }
                        if data:
                            _log.info(f"Updating dataset '{dataset_cfg['title']}' with {data}")
                            dataset = await self.update(
                                dataset.id, schemas.DataSetUpdate(**data), auth_context=auth_context
                            )
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
        vector_store_factory = VectorStoreFactory(session=self._session)

        dataset_versions: dict[uuid.UUID, int] = {
            dataset.id_: versions[dataset.id].id for dataset in datasets
        }
        data_sources: dict[uuid.UUID, int] = {
            dataset.id_: dataset.data_source_id for dataset in datasets
        }

        collections = [
            channel.available_dimensions_table_name,
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
        await self._session.commit()

        dataset = await handler.get_dataset(
            entity_id=item.id_,
            title=item.title,
            config=item.details,
            auth_context=auth_context,
            allow_offline=True,
        )
        return DataSetSerializer.db_to_schema(item, dataset)

    async def load_available_datasets(
        self, source_id: int, auth_context: AuthContext
    ) -> list[schemas.DataSetDescriptor]:
        handler = await self._get_handler(source_id)

        datasets = []
        for ds in await handler.list_datasets(auth_context):
            datasets.append(
                schemas.DataSetDescriptor(
                    data_source_id=source_id,
                    title=ds.name,
                    description=ds.description or "",
                    details=ds.details.model_dump(mode="json", by_alias=True),
                )
            )

        return datasets

    async def update(
        self, item_id: int, data: schemas.DataSetUpdate, auth_context: AuthContext
    ) -> schemas.DataSet:
        item = await self.get_model_by_id(item_id, expand=True)

        for attr, value in data.model_dump(exclude_unset=True, exclude={'details'}).items():
            setattr(item, attr, value)

        handler = await self._get_handler(item.source_id)
        if data.details is not None:
            dataset_config = await self._parse_details_field(handler, data.details)  # type: ignore
            item.details = dataset_config.model_dump(mode='json', by_alias=True)

        item.updated_at = func.now()
        await self._session.commit()
        await self._session.refresh(item)

        dataset = await handler.get_dataset(
            entity_id=item.id_,
            title=item.title,
            config=item.details,
            auth_context=auth_context,
            allow_offline=True,
        )
        return DataSetSerializer.db_to_schema(item, dataset, expand=True)

    async def delete(self, item_id: int) -> None:
        item = await self.get_model_by_id(item_id)

        count = await self.get_channel_datasets_count(dataset_id=item.id)
        if count > 0:
            _log.warning(
                f"The dataset(id={item_id}) is used in {count} channels, therefore it cannot be deleted."
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Cannot delete dataset that is used in at least one channel."
                    f" Currently {count} channels are using this dataset."
                ),
            )

        _log.info(f"Deleting dataset(id={item.id}): {item.title!r}")
        await self._session.delete(item)
        await self._session.commit()

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

    async def remove_channel_dataset(
        self, channel_id: int, dataset_id: int, auth_context: AuthContext
    ) -> None:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        dataset: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_none(
            channel_id=channel.id, dataset_id=dataset.id
        )
        if not channel_dataset:
            return

        await self._clear_channel_dataset_data(
            channel, auth_context=auth_context, dataset_id=dataset.id_, version_ids=None
        )
        await self._session.delete(channel_dataset)
        await self._session.commit()

    async def _clear_channel_dataset_data(
        self,
        channel: models.Channel,
        auth_context: AuthContext,
        *,
        dataset_id: uuid.UUID | None,
        version_ids: list[int] | None,
    ) -> None:
        """Clears all data related to a dataset or specific versions of a dataset"""
        await self._clear_vector_stores(
            channel, auth_context=auth_context, dataset_id=dataset_id, version_ids=version_ids
        )
        if ChannelService.is_channel_hybrid(channel):
            await self._clear_elastic_indices(
                channel, dataset_id=dataset_id, version_ids=version_ids
            )

    async def _clear_vector_stores(
        self,
        channel: models.Channel,
        auth_context: AuthContext,
        dataset_id: uuid.UUID | None,
        version_ids: list[int] | None,
    ) -> None:
        vector_store_factory = VectorStoreFactory(session=self._session)

        collections = [
            channel.indicator_table_name,
            channel.special_dimensions_table_name,
            channel.available_dimensions_table_name,
        ]
        for collection_name in collections:
            vector_store = await vector_store_factory.get_vector_store(
                collection_name=collection_name,
                auth_context=auth_context,
                embedding_model_name=channel.llm_model,
            )
            await vector_store.remove_documents_by(dataset_id=dataset_id, version_ids=version_ids)

    @staticmethod
    async def _clear_elastic_indices(
        channel: models.Channel, *, dataset_id: uuid.UUID | None, version_ids: list[int] | None
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
        self, channel_dataset_id: int, reason: str, preprocessing_status: StatusEnum
    ) -> models.ChannelDatasetVersion:
        item = models.ChannelDatasetVersion(
            channel_dataset_id=channel_dataset_id,
            # `version` will be set by the DB trigger automatically
            preprocessing_status=preprocessing_status,
            creation_reason=reason,
        )

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

    async def rollback_channel_dataset_to_previous_version(
        self, channel_id: int, dataset_id: int
    ) -> schemas.ChannelDatasetVersion:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        dataset: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel.id, dataset_id=dataset.id
        )

        last_version = await self._get_latest_channel_dataset_version_schema(
            channel_dataset_id=channel_dataset.id
        )
        if last_version and last_version.preprocessing_status not in StatusEnum.final_statuses():
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

    async def apply_config_to_channel_dataset(
        self, channel_id: int, dataset_id: int
    ) -> schemas.ChannelDatasetVersion:
        """Create a new version with updated config but existing indexed data.

        This allows applying non-indexing config changes (citation, pinned_columns,
        is_required, defaultQueries, etc.) without re-indexing the dataset.
        """
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        dataset: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel.id, dataset_id=dataset.id
        )

        last_version = await self._get_latest_channel_dataset_version_schema(
            channel_dataset_id=channel_dataset.id
        )
        if last_version and last_version.preprocessing_status not in StatusEnum.final_statuses():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot apply config while the indexing is in progress.",
            )

        last_completed_versions = await self._get_latest_successful_channel_dataset_versions(
            channel_dataset_ids=[channel_dataset.id]
        )
        last_completed = last_completed_versions[channel_dataset.id].last_completed_version
        if last_completed is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No completed version exists to apply config to.",
            )

        handler = await self._get_handler(dataset.source_id)

        if last_completed.resolved_config is None:
            new_resolved_config = dataset.details
        else:
            new_resolved_config = handler.merge_config_with_resolved(
                current_config=dataset.details,
                resolved_config=last_completed.resolved_config,
            )

        # Calculate new config hash from the merged config
        parsed_config = handler.parse_data_set_config(new_resolved_config)
        config_hash = parsed_config.indexing_hash

        last_indexing_hash = last_completed.indexing_config_hash
        if last_indexing_hash is not None and config_hash != last_indexing_hash:
            # This might happen if:
            # 1) the merging logic is faulty, or
            # 2) resolved_config was missing and current config has indexing-related changes.
            _log.warning(
                f"Indexing-related config has changed: {last_indexing_hash!r} -> {config_hash!r}"
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The indexing-related config has changed. Please reindex the dataset.",
            )

        new_item = models.ChannelDatasetVersion(
            channel_dataset_id=channel_dataset.id,
            # `version` will be set by the DB trigger automatically
            preprocessing_status=StatusEnum.COMPLETED,
            pointer_to=last_completed.version_data_id,
            creation_reason="Applied dataset config changes without re-indexing",
            resolved_config=new_resolved_config,
            indexing_config_hash=config_hash,
            structure_metadata=last_completed.structure_metadata,
            structure_hash=last_completed.structure_hash,
            indicator_dimensions_hash=last_completed.indicator_dimensions_hash,
            non_indicator_dimensions_hash=last_completed.non_indicator_dimensions_hash,
            special_dimensions_hash=last_completed.special_dimensions_hash,
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

    async def clear_channel_dataset_versions_data(
        self, channel_id: int, dataset_id: int, auth_context: AuthContext
    ):
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        dataset: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel_id, dataset_id=dataset.id
        )

        # We don't need to clear older versions since most likely they have been already cleared
        versions = await self.get_channel_dataset_version_models(
            limit=20, offset=0, channel_dataset_id=channel_dataset.id
        )

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

        if versions_to_clear:
            await self._clear_channel_dataset_data(
                channel, auth_context=auth_context, dataset_id=None, version_ids=versions_to_clear
            )
        else:
            _log.info("No versions to clear data for.")

    async def set_failed_status_for_channel_dataset_version(self) -> None:
        """Sets the status of all not-completed channel dataset versions to FAILED."""

        _log.info("Setting FAILED status for all non-completed channel dataset versions...")

        query = (
            update(models.ChannelDatasetVersion)
            .where(
                models.ChannelDatasetVersion.preprocessing_status.notin_(
                    StatusEnum.final_statuses()
                ),
                models.ChannelDatasetVersion.updated_at < text("NOW() - INTERVAL '12 hours'"),
            )
            .values(
                preprocessing_status=StatusEnum.FAILED,
                reason_for_failure=func.coalesce(
                    models.ChannelDatasetVersion.reason_for_failure,
                    "The version had invalid status.",
                ),
                updated_at=func.now(),
            )
        )

        result = await self._session.execute(query)
        await self._session.commit()

        _log.info(f"Updated {result.rowcount} channel dataset version(s) to FAILED status")

    async def reload_all_indicators(
        self,
        background_tasks: BackgroundTasks,
        channel_id: int,
        auth_context: AuthContext,
        max_n_embeddings: int | None = None,
    ) -> list[schemas.ChannelDatasetExpanded]:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
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
                auth_context=auth_context,
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
        return [
            ChannelDataSetSerializer.db_to_schema(
                item_db=ch_ds,
                dataset=next(ds for ds in datasets if ds.id == ch_ds.dataset_id),
                latest_version=schemas.ChannelDatasetVersion.model_validate(
                    new_versions[ch_ds.id], from_attributes=True
                ),
                last_completed_versions=last_completed_versions[ch_ds.dataset_id],
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
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
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
            auth_context=auth_context,
        )
        await self._update_channel_dataset_status(channel_dataset, StatusEnum.NOT_STARTED)

        latest_version = schemas.ChannelDatasetVersion.model_validate(version, from_attributes=True)
        last_completed_versions_mapping = (
            await self._get_latest_successful_channel_dataset_versions(
                channel_dataset_ids=[channel_dataset.id]
            )
        )
        last_completed_versions = last_completed_versions_mapping[channel_dataset.id]
        return ChannelDataSetSerializer.db_to_schema(
            channel_dataset, dataset, latest_version, last_completed_versions
        )

    @classmethod
    def _is_harmonization_supported(cls, channel: models.Channel) -> bool:
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
        db_dataset: models.DataSet,
        vector_store: VectorStore,
        version: models.ChannelDatasetVersion,
        max_n_embeddings: int | None,
        auth_context: AuthContext,
    ):
        indicators = await dataset.get_indicators(auth_context=auth_context, allow_cached=True)
        _log.info(f"Loaded {len(indicators)} indicators.")
        if max_n_embeddings:
            indicators = indicators[:max_n_embeddings]  # for debug

        documents = (
            i.to_document({IndicatorDocumentMetadataFields.DATA_SOURCE_ID: db_dataset.source_id})
            for i in indicators
        )

        await vector_store.add_documents(
            documents, dataset_id=db_dataset.id_, version_id=version.id
        )

    @staticmethod
    async def _run_hybrid_indexer(
        channel: models.Channel,
        channel_config: schemas.ChannelConfig,
        vector_store: VectorStore,
        dataset: base.DataSet,
        version: models.ChannelDatasetVersion,
        version_ids: set[int],
        harmonize_indicator: bool,
        max_n_embeddings: int | None,
        auth_context: AuthContext,
    ):
        matching_index = await ElasticSearchFactory.get_index(
            channel.matching_index_name, allow_creation=True
        )
        indicators_index = await ElasticSearchFactory.get_index(
            channel.indicators_index_name, allow_creation=True
        )
        if (data_query_config := channel_config.data_query) is None:
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
        await indexer.index(
            dataset,
            version_id=version.id,
            version_ids=version_ids,
            max_n_indicators=max_n_embeddings,
            auth_context=auth_context,
        )

    @staticmethod
    async def _index_available_dimensions(
        version: models.ChannelDatasetVersion,
        channel: models.Channel,
        dataset: base.DataSet,
        db_dataset: models.DataSet,
        vector_store_factory: VectorStoreFactory,
        max_n_embeddings: int | None,
        auth_context: AuthContext,
    ):
        vector_store = await vector_store_factory.get_vector_store(
            collection_name=channel.available_dimensions_table_name,
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
                document.metadata[field_name] = db_dataset.source_id
                documents.append(document)
        await vector_store.add_documents(
            documents, dataset_id=db_dataset.id_, version_id=version.id
        )

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
                document.metadata[field_name] = db_dataset.source_id
                field_name = SpecialDimensionValueDocumentMetadataFields.PROCESSOR_ID
                document.metadata[field_name] = processor_id
                documents.append(document)
        await vector_store.add_documents(
            documents, dataset_id=db_dataset.id_, version_id=version.id
        )

    async def _index_channel_indicators(
        self,
        channel: models.Channel,
        db_dataset: models.DataSet,
        version: models.ChannelDatasetVersion,
        version_ids: set[int] | None,
        harmonize_indicator: bool,
        max_n_embeddings: int | None,
        vector_store_factory: VectorStoreFactory,
        dataset: base.DataSet,
        auth_context: AuthContext,
    ):
        vector_store = await vector_store_factory.get_vector_store(
            collection_name=channel.indicator_table_name,
            embedding_model_name=channel.llm_model,
            auth_context=auth_context,
        )

        channel_config = schemas.ChannelConfig.model_validate(channel.details)

        if channel_config.data_query is None:
            _log.info(f"No data query found for {version}, skipping indexing")
            return

        indexer_version = channel_config.data_query.details.indexer_version
        _log.info(f"Indexer version: {indexer_version}")
        if indexer_version == schemas.IndexerVersion.hybrid:
            if version_ids is None:
                res = await self.get_latest_successful_dataset_versions_for_channel(channel.id)
                version_ids = {
                    v.last_completed_version.version_data_id
                    for v in res.values()
                    if v.last_completed_version is not None
                    and v.last_completed_version.channel_dataset_id != version.channel_dataset_id
                }

            await self._run_hybrid_indexer(
                channel=channel,
                channel_config=channel_config,
                vector_store=vector_store,
                dataset=dataset,
                version=version,
                version_ids=version_ids,
                harmonize_indicator=harmonize_indicator,
                max_n_embeddings=max_n_embeddings,
                auth_context=auth_context,
            )
        elif indexer_version == schemas.IndexerVersion.semantic:
            await self._run_semantic_indexer(
                dataset, db_dataset, vector_store, version, max_n_embeddings, auth_context
            )
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
        version = await self._get_channel_dataset_version_or_raise(channel_dataset_version_id)

        _log.info(f"Start processing {version}")
        try:
            if await self._invalid_version_status(version):
                return

            await self._update_channel_dataset_version_status(
                version, new_status=StatusEnum.IN_PROGRESS
            )

            channel_dataset = await self._get_channel_dataset_model_or_raise(
                version.channel_dataset_id
            )
            _log.info(
                f"Processing version(id={version.id}, version={version.version}) of {channel_dataset}"
            )
            channel = await ChannelService(self._session).get_model_by_id(
                channel_dataset.channel_id
            )
            db_dataset: models.DataSet = await self.get_model_by_id(channel_dataset.dataset_id)
            handler_class = await DataSourceTypeService(
                self._session
            ).get_data_source_handler_class_by_id(db_dataset.source.type_id)
            config = handler_class.parse_config(db_dataset.source.details)

            handler = handler_class(config=config)
            dataset = await handler.get_dataset(
                entity_id=db_dataset.id_,
                title=db_dataset.title,
                config=db_dataset.details,
                auth_context=auth_context,
                allow_offline=False,  # Unable to reindex offline dataset
            )

            # Extract and store resolved config from loaded dataset
            resolved_config = dataset.get_resolved_config()
            await self._set_resolved_config(version, resolved_config)

            if reindex_dimensions or (reindex_indicators and not harmonize_indicator):
                config_hash = dataset.config.indexing_hash
                structure_hash, meta = await handler.get_structure_hash_and_metadata(
                    dataset_config=db_dataset.details, auth_context=auth_context
                )
                data_hashes = await self._get_data_hashes(dataset, auth_context, allow_cached=True)
                await self._set_version_hashes_and_metadata(
                    version, config_hash, structure_hash, meta, data_hashes
                )

            vector_store_factory = VectorStoreFactory(session=self._session)

            if reindex_dimensions:
                await self._index_available_dimensions(
                    version=version,
                    channel=channel,
                    dataset=dataset,
                    db_dataset=db_dataset,
                    vector_store_factory=vector_store_factory,
                    max_n_embeddings=max_n_embeddings,
                    auth_context=auth_context,
                )

            if reindex_indicators:
                await self._index_channel_indicators(
                    channel=channel,
                    db_dataset=db_dataset,
                    version=version,
                    version_ids=version_ids,
                    harmonize_indicator=harmonize_indicator,
                    max_n_embeddings=max_n_embeddings,
                    vector_store_factory=vector_store_factory,
                    dataset=dataset,
                    auth_context=auth_context,
                )

            await self._update_channel_dataset_version_status(
                version, new_status=status_on_completion
            )
            if status_on_completion is StatusEnum.COMPLETED:
                await self._update_channel_dataset_status(
                    channel_dataset, new_status=StatusEnum.QUEUED
                )
            _log.info(f'Finished processing {version} of {channel_dataset}')
        except Exception as e:
            _log.exception(f"Failed to reindex {version}")
            await self._update_channel_dataset_version_status(
                version, new_status=StatusEnum.FAILED, reason_for_failure=str(e)
            )

    async def clear_channel_dataset_data_in_background(
        self, channel_dataset_id: int, auth_context: AuthContext
    ) -> None:
        channel_dataset = await self._get_channel_dataset_model_or_raise(channel_dataset_id)

        _log.info(f"Clear data after reindexing {channel_dataset}")
        try:
            await self._update_channel_dataset_status(
                channel_dataset, new_status=StatusEnum.IN_PROGRESS
            )

            # In case of failure, we clear the data that might have been partially indexed
            # In case of success, we clear previous version data to save space
            await self.clear_channel_dataset_versions_data(
                channel_dataset.channel_id, channel_dataset.dataset_id, auth_context
            )
            await self._update_channel_dataset_status(
                channel_dataset, new_status=StatusEnum.COMPLETED
            )
        except Exception:
            _log.exception(f"Failed to clear data after reindexing {channel_dataset}")
            await self._update_channel_dataset_status(channel_dataset, new_status=StatusEnum.FAILED)

    async def _get_deduplication_status_by_versions(
        self,
        available_dims_store: VectorStore,
        special_dims_store: VectorStore,
        indicator_dims_store: VectorStore,
        versions: set[int],
    ) -> schemas.DeduplicationStatus:
        available_has_duplicates, available_count = (
            await available_dims_store.has_duplicates_in_versions(version_ids=versions)
        )
        special_has_duplicates, special_count = await special_dims_store.has_duplicates_in_versions(
            version_ids=versions
        )
        _, indicator_count = await indicator_dims_store.has_duplicates_in_versions(
            version_ids=versions
        )

        # we only consider available and special dimensions for deduplication requirement
        deduplication_required = available_has_duplicates or special_has_duplicates
        total_duplicates = available_count + special_count + indicator_count

        return schemas.DeduplicationStatus(
            deduplication_required=deduplication_required,
            total_duplicate_count=total_duplicates,
            available_dimensions_duplicate_count=available_count,
            special_dimensions_duplicate_count=special_count,
            indicator_dimensions_duplicate_count=indicator_count,
        )

    async def _get_full_deduplication_status(
        self,
        available_dims_store: VectorStore,
        special_dims_store: VectorStore,
        indicator_dims_store: VectorStore,
    ) -> schemas.DeduplicationStatus:
        available_has_duplicates, available_count = await available_dims_store.has_duplicates()
        special_has_duplicates, special_count = await special_dims_store.has_duplicates()
        _, indicator_count = await indicator_dims_store.has_duplicates()

        # we only consider available and special dimensions for deduplication requirement
        deduplication_required = available_has_duplicates or special_has_duplicates
        total_duplicates = available_count + special_count + indicator_count

        return schemas.DeduplicationStatus(
            deduplication_required=deduplication_required,
            total_duplicate_count=total_duplicates,
            available_dimensions_duplicate_count=available_count,
            special_dimensions_duplicate_count=special_count,
            indicator_dimensions_duplicate_count=indicator_count,
        )

    async def _check_latest_versions_status(
        self,
        channel: models.Channel,
        available_dims_store: VectorStore,
        special_dims_store: VectorStore,
        indicator_dims_store: VectorStore,
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
            available_dims_store,
            special_dims_store,
            indicator_dims_store,
            versions,
        )
        sizes = schemas.VectorStoreSizes(
            available_dimensions_size=await available_dims_store.get_size(version_ids=versions),
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
        available_dims_store: VectorStore,
        special_dims_store: VectorStore,
        indicator_dims_store: VectorStore,
    ) -> schemas.ChannelIndexStatus:
        deduplication_status = await self._get_full_deduplication_status(
            available_dims_store,
            special_dims_store,
            indicator_dims_store,
        )
        sizes = schemas.VectorStoreSizes(
            available_dimensions_size=await available_dims_store.get_total_size(),
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
        auth_context: AuthContext,
        scope: schemas.ChannelIndexStatusScope,
    ) -> schemas.ChannelIndexStatus:
        """Checks index status for channel"""
        channel = await ChannelService(self._session).get_model_by_id(channel_id)
        vector_store_factory = VectorStoreFactory(session=self._session)

        available_dims_store = await vector_store_factory.get_vector_store(
            collection_name=channel.available_dimensions_table_name,
            auth_context=auth_context,
            embedding_model_name=channel.llm_model,
        )
        special_dims_store = await vector_store_factory.get_vector_store(
            collection_name=channel.special_dimensions_table_name,
            auth_context=auth_context,
            embedding_model_name=channel.llm_model,
        )
        indicator_dims_store = await vector_store_factory.get_vector_store(
            collection_name=channel.indicator_table_name,
            auth_context=auth_context,
            embedding_model_name=channel.llm_model,
        )

        if scope == schemas.ChannelIndexStatusScope.FULL:
            return await self._check_full_index_status(
                channel=channel,
                available_dims_store=available_dims_store,
                special_dims_store=special_dims_store,
                indicator_dims_store=indicator_dims_store,
            )
        elif scope == schemas.ChannelIndexStatusScope.LATEST_COMPLETED_VERSIONS:
            return await self._check_latest_versions_status(
                channel=channel,
                available_dims_store=available_dims_store,
                special_dims_store=special_dims_store,
                indicator_dims_store=indicator_dims_store,
            )
        else:
            raise ValueError(f"Unknown scope: {scope}")


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
        async with models.get_session_contex_manager() as session:
            service = AdminPortalDataSetService(session)
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
async def clear_channel_dataset_data_in_background_task(
    channel_dataset_id: int, auth_context: AuthContext
) -> None:
    try:
        async with models.get_session_contex_manager() as session:
            service = AdminPortalDataSetService(session)
            await service.clear_channel_dataset_data_in_background(
                channel_dataset_id=channel_dataset_id,
                auth_context=auth_context,
            )
    except Exception as e:
        _log.exception(e)

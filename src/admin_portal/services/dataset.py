import csv
import io
import json
import os.path
import zipfile
from collections.abc import Iterable
from typing import Any

import yaml
from fastapi import BackgroundTasks, HTTPException, status
from pydantic import ValidationError
from sqlalchemy import delete
from sqlalchemy.sql.expression import func

import common.models as models
import common.schemas as schemas
from admin_portal.auth.auth_context import SystemUserAuthContext
from admin_portal.config import JobsConfig
from common import utils
from common.auth.auth_context import AuthContext
from common.config import (
    DimensionValueDocumentMetadataFields,
    IndicatorDocumentMetadataFields,
    VectorStoreMetadataFields,
)
from common.config import multiline_logger as logger
from common.data import base
from common.data.base.dataset import DataSetConfigType
from common.indexer import IndexerFactory
from common.schemas import PreprocessingStatusEnum as StatusEnum
from common.services import DataSetSerializer, DataSetService
from common.utils.elastic import ElasticIndex, ElasticSearchFactory, SearchResult
from common.vectorstore import EmbeddedDocument, VectorStore, VectorStoreFactory

from .background_tasks import background_task
from .channel import AdminPortalChannelService as ChannelService
from .data_source import AdminPortalDataSourceService as DataSourceService
from .data_source import DataSourceTypeService


class AdminPortalDataSetService(DataSetService):
    async def _export_vector_store_data(
        self,
        channel: models.Channel,
        datasets: Iterable[schemas.DataSet],
        res_dir: str,
        auth_context: AuthContext,
    ) -> None:
        logger.info("Exporting vector store data...")
        vector_store_factory = VectorStoreFactory(session=self._session)

        for table in [channel.available_dimensions_table_name, channel.indicator_table_name]:
            vector_store = vector_store_factory.get_vector_store(
                collection_name=table,
                auth_context=auth_context,
                embedding_model_name=channel.llm_model,
            )

            vector_store_folder = os.path.join(res_dir, table.split('_', maxsplit=1)[0])
            os.makedirs(vector_store_folder, exist_ok=True)

            for dataset in datasets:
                logger.info(f"Exporting {table} (dataset: {dataset.title})...")
                file_name = utils.escape_invalid_filename_chars(
                    f"{dataset.data_source.title}_{dataset.title}.csv"
                )
                file_path = os.path.join(vector_store_folder, file_name)
                logger.info(f"Saving to {file_path}")

                with open(file_path, 'w', encoding=JobsConfig.ENCODING, newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(JobsConfig.CSV_COLUMNS)

                    documents = await vector_store.get_documents(
                        dataset_id=dataset.id, include_embeddings=True
                    )

                    rows: list[tuple] = []
                    for doc in documents:
                        metadata = {
                            k: v
                            for k, v in doc.metadata.items()
                            if k not in VectorStoreMetadataFields.__ALL__
                        }
                        rows.append((doc.page_content, json.dumps(metadata), doc.embeddings))
                    csv_writer.writerows(rows)
                logger.info(f"Exported {table} (dataset: {dataset.title})")
        logger.info("Finished exporting vector store data")

    async def _export_elastic_data(
        self, channel: models.Channel, datasets: Iterable[schemas.DataSet], res_dir: str
    ) -> None:
        async def _es_get_all(index: ElasticIndex, dataset_id: str) -> SearchResult:
            query = {
                "term": {"dataset_id": dataset_id},
            }
            res = await index.search(query=query, size=10000)

            if res.hits.total.value > 10000:
                # Our indexes contain less than 10 thousand documents, but added a check just in case
                raise RuntimeError(f"Too many documents to export: {res.hits.total.value}")

            return res

        logger.info("Exporting elastic data...")

        matching_index = await ElasticSearchFactory.get_index(channel.matching_index_name)
        indicators_index = await ElasticSearchFactory.get_index(channel.indicators_index_name)

        indexes = [
            (JobsConfig.ES_MATCHING_DIR, matching_index),
            (JobsConfig.ES_INDICATORS_DIR, indicators_index),
        ]

        for folder, index in indexes:
            index_folder = os.path.join(res_dir, folder)
            os.makedirs(index_folder, exist_ok=True)

            for dataset in datasets:
                res = await _es_get_all(index, str(dataset.id))
                documents = [hit.source for hit in res.hits.hits]

                file_name = utils.escape_invalid_filename_chars(f"{dataset.title}.jsonl")
                file_path = os.path.join(index_folder, file_name)

                for d in documents:
                    # append to jsonl file
                    utils.write_json(
                        obj=d,
                        fp=file_path,
                        mode='a+',
                        encoding=JobsConfig.ENCODING,
                        indent=None,
                        add_newline=True,
                    )
        logger.info("Finished exporting elastic data")

    async def export_datasets(
        self, channel: models.Channel, res_dir: str, auth_context: AuthContext
    ) -> None:
        channel_config = schemas.ChannelConfig.model_validate(channel.details)

        datasets = await self.get_datasets_schemas(
            limit=None,
            offset=0,
            channel_id=channel.id,
            auth_context=auth_context,
            allow_offline=True,
        )
        data_sources = {}

        data = []
        for dataset in datasets:
            dataset_json = dataset.model_dump(mode='json', include=JobsConfig.DATASET_FIELDS)
            dataset_json['dataSource'] = dataset.data_source.title
            data.append(dataset_json)

            if dataset.data_source_id not in data_sources:
                data_sources[dataset.data_source_id] = dataset.data_source

        # sort datasets by title
        data.sort(key=lambda x: x['title'])

        datasets_file = os.path.join(res_dir, JobsConfig.DATASETS_FILE)
        utils.write_yaml({'dataSets': data}, datasets_file)

        await DataSourceService.export_data_sources(data_sources.values(), res_dir)

        await self._export_vector_store_data(channel, datasets, res_dir, auth_context)

        indexer_version = channel_config.data_query.details.indexer_version
        logger.info(f"Indexer version: {indexer_version}")
        if indexer_version == schemas.IndexerVersion.hybrid:
            await self._export_elastic_data(channel, datasets, res_dir)
        else:
            logger.info("Skipping exporting elastic data")

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
                            logger.info(f"Updating dataset '{dataset_cfg['title']}' with {data}")
                            dataset = await self.update(
                                dataset.id, schemas.DataSetUpdate(**data), auth_context=auth_context
                            )
                        else:
                            logger.info(
                                f"Dataset '{dataset_cfg['title']}' exists and is up to date"
                            )
                    else:
                        logger.info(f"Dataset '{dataset_cfg['title']}' already exists. Skipping.")
                else:
                    dataset = await self.create_dataset(parsed_dataset, auth_context=auth_context)
                    dataset.data_source = data_source
                    logger.info(f"Created dataset {dataset.title}")

                datasets.append(dataset)

        return datasets

    async def _add_datasets_to_channel(
        self, channel_id: int, datasets: list[schemas.DataSet], preprocessing_status: StatusEnum
    ) -> list[models.ChannelDataset]:
        items = [
            models.ChannelDataset(
                channel_id=channel_id,
                dataset_id=ds.id,
                preprocessing_status=preprocessing_status,
            )
            for ds in datasets
        ]

        self._session.add_all(items)
        await self._session.commit()
        return items

    async def _import_vector_store_tables(
        self,
        zip_file: zipfile.ZipFile,
        channel: models.Channel,
        datasets: list[schemas.DataSet],
        auth_context: AuthContext,
    ) -> None:
        logger.info("Importing vector store data...")
        vector_store_factory = VectorStoreFactory(session=self._session)

        for table in [channel.available_dimensions_table_name, channel.indicator_table_name]:
            vector_store = vector_store_factory.get_vector_store(
                collection_name=table,
                embedding_model_name=channel.llm_model,
                auth_context=auth_context,
            )
            table_folder = table.split('_', maxsplit=1)[0]

            for dataset in datasets:
                logger.info(f"Importing {table} (dataset: {dataset.title})...")
                file_name = utils.escape_invalid_filename_chars(
                    f"{dataset.data_source.title}_{dataset.title}.csv"
                )
                file_path = os.path.join(table_folder, file_name)
                logger.info(f"Opening {file_path}")

                with zip_file.open(file_path) as csv_file:
                    text_stream = io.TextIOWrapper(csv_file, encoding=JobsConfig.ENCODING)
                    csv_reader = csv.reader(text_stream)
                    iterator = iter(csv_reader)

                    headers = next(iterator)  # skip header
                    if headers != JobsConfig.CSV_COLUMNS:
                        logger.error(f"Unexpected headers: {headers} in {file_path}")
                        continue

                    documents = [
                        EmbeddedDocument(
                            page_content=page_content,
                            metadata=json.loads(metadata),
                            embeddings=json.loads(embeddings),
                        )
                        for page_content, metadata, embeddings in iterator
                    ]

                    if documents:
                        field_name = DimensionValueDocumentMetadataFields.DATA_SOURCE_ID

                        if documents[0].metadata.get(field_name) != dataset.data_source_id:
                            # Fix data source id in metadata:
                            for document in documents:
                                document.metadata[field_name] = dataset.data_source_id

                        await vector_store.import_documents(documents, dataset_id=dataset.id)
                    else:
                        logger.warning(f"No documents found in {file_path}")
                logger.info(f"Imported {table} (dataset: {dataset.title})")
        logger.info("Finished importing vector store data")
        logger.info('-' * 40)

    async def _import_elastic_data(
        self, zip_file: zipfile.ZipFile, channel: models.Channel, datasets: list[schemas.DataSet]
    ) -> None:
        logger.info("Importing elastic data...")

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
                file_name = utils.escape_invalid_filename_chars(f"{dataset.title}.jsonl")
                file_path = os.path.join(folder, file_name)

                if file_path not in zip_file.namelist():
                    logger.warning(f"File '{file_path}' not found in the zip archive")
                    continue
                logger.info(f"Opening '{file_path}'")

                with zip_file.open(file_path) as file:
                    documents: list[dict[str, str]] = []
                    for line in file.readlines():
                        doc = json.loads(line)

                        # Fix dataset metadata:
                        doc['dataset_id'] = str(dataset.id)
                        doc['dataset_name'] = dataset.title

                        documents.append(doc)

                await index.add_bulk(documents)
        logger.info("Finished importing elastic data")
        logger.info('-' * 40)

    async def import_datasets_and_data_sources_from_zip(
        self,
        channel_db: models.Channel,
        zip_file: zipfile.ZipFile,
        update_datasets: bool,
        update_data_sources: bool,
        auth_context: AuthContext,
    ) -> None:
        channel_config = schemas.ChannelConfig.model_validate(channel_db.details)

        source_service = DataSourceService(self._session)
        data_sources = await source_service.import_data_sources_from_zip(
            zip_file, update_data_sources
        )

        datasets = await self._import_datasets(
            zip_file, data_sources, update_datasets, auth_context=auth_context
        )
        channel_datasets = await self._add_datasets_to_channel(
            channel_id=channel_db.id, datasets=datasets, preprocessing_status=StatusEnum.QUEUED
        )

        await self._import_vector_store_tables(zip_file, channel_db, datasets, auth_context)

        indexer_version = channel_config.data_query.details.indexer_version
        logger.info(f"Indexer version: {indexer_version}")
        if indexer_version == schemas.IndexerVersion.hybrid:
            await self._import_elastic_data(zip_file, channel_db, datasets)
        else:
            logger.info("Skipping importing elastic data")

        for channel_dataset in channel_datasets:
            channel_dataset.preprocessing_status = StatusEnum.COMPLETED
        await self._session.commit()

    async def _parse_details_field(
        self, handler: base.DataSourceHandler, details: dict[str, Any]
    ) -> DataSetConfigType:
        try:
            parsed_config = handler.parse_data_set_config(details)
        except ValidationError as e:
            logger.info(e)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to parse 'details' field: {e}",
            )
        except Exception as e:
            logger.info(e)
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
            entity_id=str(item.id),
            title=item.title,
            config=item.details,
            auth_context=auth_context,
            allow_offline=True,
        )
        return DataSetSerializer.db_to_schema(item, dataset)

    async def load_available_datasets(self, source_id: int) -> list[schemas.DataSetDescriptor]:
        handler = await self._get_handler(source_id)

        datasets = []
        for ds in await handler.list_datasets(SystemUserAuthContext()):
            datasets.append(
                schemas.DataSetDescriptor(
                    data_source_id=source_id,
                    title=ds.name,
                    description=ds.description or "",
                    details=ds.details,
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
            entity_id=str(item.id),
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
            logger.warning(
                f"The dataset(id={item_id}) is used in {count} channels, therefore it cannot be deleted."
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Cannot delete dataset that is used in at least one channel."
                    f" Currently {count} channels are using this dataset."
                ),
            )

        logger.info(f"Deleting dataset(id={item.id}): {item.title!r}")
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
            preprocessing_status=StatusEnum.NOT_STARTED,
        )

        self._session.add(item)
        await self._session.commit()

        return schemas.ChannelDatasetBase.model_validate(item, from_attributes=True)

    async def remove_channel_dataset(
        self, channel_id: int, dataset_id: int, auth_context: AuthContext
    ):
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        dataset: models.DataSet = await self.get_model_by_id(dataset_id)

        query = (
            delete(models.ChannelDataset)
            .where(models.ChannelDataset.channel_id == channel.id)
            .where(models.ChannelDataset.dataset_id == dataset.id)
        )
        await self._session.execute(query)
        await self._session.commit()

        await self._clear_indicators(channel, dataset, auth_context=auth_context)

    async def _clear_indicators(
        self, channel: models.Channel, dataset: models.DataSet, auth_context: AuthContext
    ):
        vector_store_factory = VectorStoreFactory(session=self._session)
        vector_store = vector_store_factory.get_vector_store(
            collection_name=channel.indicator_table_name,
            auth_context=auth_context,
            embedding_model_name=channel.llm_model,
        )

        await vector_store.remove_documents_by_dataset_id(dataset.id)

        # TODO: clear available dimensions as well

    async def _update_channel_dataset_status(
        self, item: models.ChannelDataset, new_status: StatusEnum
    ):
        item.preprocessing_status = new_status
        item.updated_at = func.now()
        await self._session.commit()
        await self._session.refresh(item)

    async def reload_all_indicators(
        self,
        background_tasks: BackgroundTasks,
        channel_id: int,
        reindex_indicators: bool,
        harmonize_indicator: bool | None,
        reindex_dimensions: bool,
        auth_context: AuthContext,
        max_n_embeddings: int | None = None,
    ) -> list[schemas.ChannelDatasetBase]:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        channel_datasets: list[models.ChannelDataset] = await self.get_channel_dataset_models(
            limit=None, offset=0, channel_id=channel.id
        )
        harmonization_supported = self._is_harmonization_supported(channel)
        normalization_required = (
            not harmonization_supported or harmonize_indicator is None or not harmonize_indicator
        )
        harmonization_required = harmonization_supported and (
            harmonize_indicator is None or harmonize_indicator
        )

        if normalization_required:
            status_on_completion = (
                StatusEnum.QUEUED if harmonization_required else StatusEnum.COMPLETED
            )
            for ch_ds in channel_datasets:
                previous_status = ch_ds.preprocessing_status

                if previous_status in (StatusEnum.QUEUED, StatusEnum.IN_PROGRESS):
                    logger.warning(
                        f"{channel.id=} dataset_id={ch_ds.dataset_id} "
                        "Pre-processing of the channel dataset is already in progress."
                    )
                    continue

                await self._update_channel_dataset_status(ch_ds, new_status=StatusEnum.QUEUED)

                background_tasks.add_task(
                    reload_indicators_in_background_task,
                    channel_id=channel.id,
                    dataset_id=ch_ds.dataset_id,
                    reindex_indicators=reindex_indicators,
                    harmonize_indicator=False,
                    reindex_dimensions=reindex_dimensions,
                    auth_context=auth_context,
                    previous_status=previous_status,
                    max_n_embeddings=max_n_embeddings,
                    status_on_completion=status_on_completion,
                )
        if harmonization_required:
            for ch_ds in channel_datasets:
                previous_status = ch_ds.preprocessing_status

                if (
                    previous_status in (StatusEnum.QUEUED, StatusEnum.IN_PROGRESS)
                    and not normalization_required
                ):
                    logger.warning(
                        f"{channel.id=} dataset_id={ch_ds.dataset_id} "
                        "Pre-processing of the channel dataset is already in progress."
                    )
                    continue

                if not normalization_required:
                    # it's required to update the status only if normalization was not planned
                    await self._update_channel_dataset_status(ch_ds, new_status=StatusEnum.QUEUED)

                background_tasks.add_task(
                    reload_indicators_in_background_task,
                    channel_id=channel.id,
                    dataset_id=ch_ds.dataset_id,
                    reindex_indicators=reindex_indicators,
                    harmonize_indicator=True,
                    # do not reindex dimensions if it's already been done
                    reindex_dimensions=reindex_indicators and not normalization_required,
                    auth_context=auth_context,
                    previous_status=previous_status,
                    max_n_embeddings=max_n_embeddings,
                )

        return [
            schemas.ChannelDatasetBase.model_validate(ch_ds, from_attributes=True)
            for ch_ds in channel_datasets
        ]

    async def reload_indicators(
        self,
        background_tasks: BackgroundTasks,
        channel_id: int,
        dataset_id: int,
        reindex_indicators: bool,
        harmonize_indicator: bool | None,
        reindex_dimensions: bool,
        auth_context: AuthContext,
        max_n_embeddings: int | None = None,
    ) -> schemas.ChannelDatasetBase:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        db_dataset: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel_id, dataset_id=dataset_id
        )
        previous_status = channel_dataset.preprocessing_status

        if previous_status in (StatusEnum.QUEUED, StatusEnum.IN_PROGRESS):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Pre-processing of the channel dataset is already in progress.",
            )

        await self._update_channel_dataset_status(channel_dataset, new_status=StatusEnum.QUEUED)

        harmonization_supported = self._is_harmonization_supported(channel)
        normalization_required = (
            not harmonization_supported or harmonize_indicator is None or not harmonize_indicator
        )
        harmonization_required = harmonization_supported and (
            harmonize_indicator is None or harmonize_indicator
        )

        if normalization_required:
            background_tasks.add_task(
                reload_indicators_in_background_task,
                channel_id=channel.id,
                dataset_id=db_dataset.id,
                reindex_indicators=reindex_indicators,
                harmonize_indicator=False,
                reindex_dimensions=reindex_dimensions,
                auth_context=auth_context,
                previous_status=previous_status,
                max_n_embeddings=max_n_embeddings,
                status_on_completion=(
                    StatusEnum.QUEUED if harmonization_required else StatusEnum.COMPLETED
                ),
            )

        if harmonization_required:
            background_tasks.add_task(
                reload_indicators_in_background_task,
                channel_id=channel.id,
                dataset_id=db_dataset.id,
                # do not reindex dimensions if it's already been done
                reindex_indicators=reindex_indicators and not normalization_required,
                harmonize_indicator=True,
                reindex_dimensions=reindex_dimensions,
                auth_context=auth_context,
                previous_status=previous_status,
                max_n_embeddings=max_n_embeddings,
            )
        return schemas.ChannelDatasetBase.model_validate(channel_dataset, from_attributes=True)

    @staticmethod
    def _is_harmonization_supported(channel: models.Channel) -> bool:
        channel_config = schemas.ChannelConfig.model_validate(channel.details)
        indexer_version = channel_config.data_query.details.indexer_version
        return indexer_version == schemas.IndexerVersion.hybrid

    @staticmethod
    async def _run_semantic_indexer(
        dataset: base.DataSet,
        db_dataset: models.DataSet,
        vector_store: VectorStore,
        max_n_embeddings: int | None,
        auth_context: AuthContext,
    ):
        indicators = await dataset.get_indicators(auth_context=auth_context)
        logger.info(f"Loaded {len(indicators)} indicators.")
        if max_n_embeddings:
            indicators = indicators[:max_n_embeddings]  # for debug

        # documents: list[Document] = [indicator.to_document() for indicator in indicators]
        # for doc in documents:
        #     doc.metadata[IndicatorDocumentMetadataFields.DATA_SOURCE_ID] = db_dataset.source_id

        documents = (
            i.to_document({IndicatorDocumentMetadataFields.DATA_SOURCE_ID: db_dataset.source_id})
            for i in indicators
        )

        await vector_store.add_documents(documents, dataset_id=db_dataset.id)

    @staticmethod
    async def _run_hybrid_indexer(
        channel: models.Channel,
        vector_store: VectorStore,
        dataset: base.DataSet,
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

        indexer = IndexerFactory(
            auth_context.api_key, matching_index, indicators_index, vector_store
        ).get_indexer(normalize=not harmonize_indicator, harmonize=harmonize_indicator)
        await indexer.index(dataset, max_n_indicators=max_n_embeddings, auth_context=auth_context)

    @staticmethod
    async def _index_available_dimensions(
        channel: models.Channel,
        dataset: base.DataSet,
        db_dataset: models.DataSet,
        channel_dataset: models.ChannelDataset,
        previous_status: StatusEnum,
        vector_store_factory: VectorStoreFactory,
        auth_context: AuthContext,
    ):
        vector_store = vector_store_factory.get_vector_store(
            collection_name=channel.available_dimensions_table_name,
            embedding_model_name=channel.llm_model,
            auth_context=auth_context,
        )

        if previous_status in [StatusEnum.COMPLETED, StatusEnum.FAILED]:
            logger.info(f"Clear existing data for the {channel_dataset}")
            await vector_store.remove_documents_by_dataset_id(db_dataset.id)

        dimensions = dataset.non_indicator_dimensions()
        documents = []
        for dimension in dimensions:
            if not isinstance(dimension, base.CategoricalDimension):
                continue
            category_values = dimension.available_values
            for value in category_values:
                document = value.to_document()
                field_name = DimensionValueDocumentMetadataFields.DATA_SOURCE_ID
                document.metadata[field_name] = db_dataset.source_id
                documents.append(document)
        await vector_store.add_documents(documents, dataset_id=db_dataset.id)

    async def _index_channel_indicators(
        self,
        channel: models.Channel,
        db_dataset: models.DataSet,
        channel_dataset: models.ChannelDataset,
        previous_status: StatusEnum,
        harmonize_indicator: bool,
        max_n_embeddings: int | None,
        vector_store_factory: VectorStoreFactory,
        dataset: base.DataSet,
        auth_context: AuthContext,
    ):
        vector_store = vector_store_factory.get_vector_store(
            collection_name=channel.indicator_table_name,
            embedding_model_name=channel.llm_model,
            auth_context=auth_context,
        )

        if not harmonize_indicator and previous_status in [StatusEnum.COMPLETED, StatusEnum.FAILED]:
            logger.info(f"Clear existing indicators for the {channel_dataset}")
            await vector_store.remove_documents_by_dataset_id(db_dataset.id)

        channel_config = schemas.ChannelConfig.model_validate(channel.details)
        indexer_version = channel_config.data_query.details.indexer_version
        logger.info(f"Indexer version: {indexer_version}")
        if indexer_version == schemas.IndexerVersion.hybrid:
            await self._run_hybrid_indexer(
                channel, vector_store, dataset, harmonize_indicator, max_n_embeddings, auth_context
            )
        elif indexer_version == schemas.IndexerVersion.semantic:
            await self._run_semantic_indexer(
                dataset, db_dataset, vector_store, max_n_embeddings, auth_context
            )
        else:
            raise RuntimeError(f"Unknown indexer version: {indexer_version}")

    async def reload_channel_dataset_in_background(
        self,
        channel_id: int,
        dataset_id: int,
        reindex_indicators: bool,
        harmonize_indicator: bool,
        reindex_dimensions: bool,
        auth_context: AuthContext,
        previous_status: StatusEnum,
        max_n_embeddings: int | None,
        status_on_completion: StatusEnum = StatusEnum.COMPLETED,
    ) -> None:
        channel: models.Channel = await ChannelService(self._session).get_model_by_id(channel_id)
        db_dataset: models.DataSet = await self.get_model_by_id(dataset_id)
        channel_dataset = await self.get_channel_dataset_model_or_raise(
            channel_id=channel_id, dataset_id=dataset_id
        )
        handler_class = await DataSourceTypeService(
            self._session
        ).get_data_source_handler_class_by_id(db_dataset.source.type_id)
        config = handler_class.parse_config(db_dataset.source.details)

        logger.info(f"Start processing {channel_dataset}")
        try:
            await self._update_channel_dataset_status(
                channel_dataset, new_status=StatusEnum.IN_PROGRESS
            )

            dataset = await handler_class(config=config).get_dataset(
                entity_id=str(db_dataset.id),
                title=db_dataset.title,
                config=db_dataset.details,
                auth_context=auth_context,
                allow_offline=False,  # Unable to reindex offline dataset
            )

            vector_store_factory = VectorStoreFactory(session=self._session)

            if reindex_dimensions:
                await self._index_available_dimensions(
                    channel=channel,
                    dataset=dataset,
                    db_dataset=db_dataset,
                    channel_dataset=channel_dataset,
                    previous_status=previous_status,
                    vector_store_factory=vector_store_factory,
                    auth_context=auth_context,
                )

            if reindex_indicators:
                await self._index_channel_indicators(
                    channel=channel,
                    db_dataset=db_dataset,
                    channel_dataset=channel_dataset,
                    previous_status=previous_status,
                    harmonize_indicator=harmonize_indicator,
                    max_n_embeddings=max_n_embeddings,
                    vector_store_factory=vector_store_factory,
                    dataset=dataset,
                    auth_context=auth_context,
                )

            await self._update_channel_dataset_status(
                channel_dataset, new_status=status_on_completion
            )
        except Exception as e:
            logger.exception(e)
            await self._update_channel_dataset_status(channel_dataset, new_status=StatusEnum.FAILED)
        finally:
            logger.info(f'Finished processing {channel_dataset}')


@background_task
async def reload_indicators_in_background_task(
    channel_id: int,
    dataset_id: int,
    reindex_indicators: bool,
    harmonize_indicator: bool,
    reindex_dimensions: bool,
    auth_context: AuthContext,
    previous_status: StatusEnum,
    max_n_embeddings: int | None,
    status_on_completion: StatusEnum = StatusEnum.COMPLETED,
) -> None:
    try:
        async with models.get_session_contex_manager() as session:
            service = AdminPortalDataSetService(session)
            await service.reload_channel_dataset_in_background(
                channel_id=channel_id,
                dataset_id=dataset_id,
                reindex_indicators=reindex_indicators,
                harmonize_indicator=harmonize_indicator,
                reindex_dimensions=reindex_dimensions,
                auth_context=auth_context,
                previous_status=previous_status,
                max_n_embeddings=max_n_embeddings,
                status_on_completion=status_on_completion,
            )
    except Exception as e:
        logger.exception(e)

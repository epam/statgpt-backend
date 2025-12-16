import asyncio
import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import NamedTuple

from fastapi import HTTPException, status
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.sql.expression import func

import statgpt.common.models as models
import statgpt.common.schemas as schemas
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import DataSet, DataSourceHandler
from statgpt.common.schemas import PreprocessingStatusEnum as StatusEnum
from statgpt.common.settings.dataflow_loader import DataflowLoaderSettings
from statgpt.common.utils import async_utils

from .base import DbServiceBase
from .data_source import DataSourceSerializer, DataSourceService, DataSourceTypeService

_log = logging.getLogger(__name__)


class LastCompletedVersions(NamedTuple):
    last_completed_version: schemas.ChannelDatasetVersion | None
    previous_completed_version: schemas.ChannelDatasetVersion | None


class DataSetSerializer:
    @staticmethod
    def db_to_schema(
        item_db: models.DataSet, dataset: DataSet, expand: bool = False
    ) -> schemas.DataSet:
        res = schemas.DataSet(
            id=item_db.id,
            id_=item_db.id_,
            created_at=item_db.created_at,
            updated_at=item_db.updated_at,
            data_source_id=item_db.source_id,
            data_source=None,
            title=dataset.name,
            description=dataset.description or "",
            details=item_db.details,
            status=dataset.status,
        )

        if expand:
            res.data_source = DataSourceSerializer.db_to_schema(item_db.source)

        return res


class ChannelDataSetSerializer:
    @staticmethod
    def db_to_schema(
        item_db: models.ChannelDataset,
        dataset: schemas.DataSet,
        latest_version: schemas.ChannelDatasetVersion | None,
        last_completed_versions: LastCompletedVersions,
    ) -> schemas.ChannelDatasetExpanded:
        preprocessing_status = (
            StatusEnum.NOT_STARTED
            if latest_version is None
            else latest_version.preprocessing_status
        )

        return schemas.ChannelDatasetExpanded(
            id=item_db.id,
            created_at=item_db.created_at,
            updated_at=item_db.updated_at,
            channel_id=item_db.channel_id,
            dataset_id=item_db.dataset_id,
            preprocessing_status=preprocessing_status,
            clearing_status=item_db.clearing_status,
            dataset=dataset,
            latest_version=latest_version,
            last_completed_version=last_completed_versions.last_completed_version,
            previous_completed_version=last_completed_versions.previous_completed_version,
        )


class DataSetService(DbServiceBase):
    _SETTINGS = DataflowLoaderSettings()

    def __init__(self, session: AsyncSession, session_lock: asyncio.Lock | None = None) -> None:
        super().__init__(session, session_lock)

    @staticmethod
    def _apply_filters(
        query: Select, source_id: int | None, channel_id: int | None, ids: Iterable[int] | None
    ) -> Select:
        if channel_id is not None:
            query = query.join(models.DataSet.mapped_channels).where(
                models.ChannelDataset.channel_id == channel_id
            )

        if source_id is not None:
            query = query.where(models.DataSet.source_id == source_id)

        if ids is not None:
            query = query.where(models.DataSet.id.in_(ids))

        return query

    async def get_datasets_count(
        self, source_id: int | None, channel_id: int | None, ids: list[int] | None = None
    ) -> int:
        query = select(func.count("*")).select_from(models.DataSet)  # type: ignore
        query = self._apply_filters(query, source_id=source_id, channel_id=channel_id, ids=ids)
        async with self._lock_session() as session:
            return (await session.execute(query)).scalar_one()

    async def get_datasets_models(
        self,
        *,
        limit: int | None,
        offset: int,
        expand: bool = False,
        source_id: int | None = None,
        channel_id: int | None = None,
        ids: Iterable[int] | None = None,
    ) -> list[models.DataSet]:
        query = select(models.DataSet)
        if expand:
            query = query.options(
                selectinload(models.DataSet.source).selectinload(models.DataSource.type)
            )

        query = self._apply_filters(query, source_id=source_id, channel_id=channel_id, ids=ids)

        async with self._lock_session() as session:
            q_result = await session.scalars(query.limit(limit).offset(offset))
        return [item for item in q_result.all()]

    async def get_datasets_schemas(
        self,
        limit: int | None,
        offset: int,
        auth_context: AuthContext,
        source_id: int | None = None,
        channel_id: int | None = None,
        ids: Iterable[int] | None = None,
        allow_offline: bool = False,
        allow_cached_datasets: bool = False,
    ) -> list[schemas.DataSet]:
        items = await self.get_datasets_models(
            limit=limit,
            offset=offset,
            expand=True,
            source_id=source_id,
            channel_id=channel_id,
            ids=ids,
        )
        sources: set[int] = {i.source_id for i in items}
        handlers = {source_id: await self._get_handler(source_id) for source_id in sources}

        tasks = []
        for item in items:
            handler = handlers[item.source_id]
            tasks.append(
                handler.get_dataset(
                    entity_id=item.id_,
                    title=item.title,
                    config=item.details,
                    auth_context=auth_context,
                    allow_offline=allow_offline,
                    allow_cached=allow_cached_datasets,
                )
            )

        datasets: list[DataSet] = await async_utils.gather_with_concurrency(
            self._SETTINGS.dataset_concurrency_limit, *tasks
        )

        return [
            DataSetSerializer.db_to_schema(item, ds, expand=True)
            for item, ds in zip(items, datasets)
        ]

    async def _get_handler(self, data_source_id: int) -> DataSourceHandler:
        source_service = DataSourceService(self._session, session_lock=self._session_lock)
        source_type_service = DataSourceTypeService(self._session, session_lock=self._session_lock)

        source: models.DataSource = await source_service.get_by_id(data_source_id)
        handler_class = await source_type_service.get_data_source_handler_class_by_id(
            source.type_id
        )

        config = handler_class.parse_config(source.details)
        _log.info(f"{config=}")

        return handler_class(config=config)

    async def _get_item_or_raise(self, item_id: int, expand: bool = False) -> models.DataSet:
        """Retrieve a models.DataSet by id or raise a 404 error if not found."""
        options = None
        if expand:
            options = [selectinload(models.DataSet.source).selectinload(models.DataSource.type)]

        async with self._lock_session() as session:
            item: models.DataSet | None = await session.get(
                models.DataSet, item_id, options=options
            )
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DataSet with id={item_id} not found",
            )
        return item

    async def get_model_by_id(self, item_id: int, expand: bool = False) -> models.DataSet:
        item = await self._get_item_or_raise(item_id, expand=expand)

        async with self._lock_session() as session:
            await session.refresh(item, attribute_names=["source"])
        return item

    async def get_schema_by_id(
        self, item_id: int, auth_context: AuthContext, allow_offline: bool = False
    ) -> schemas.DataSet:
        item = await self.get_model_by_id(item_id, expand=True)

        handler = await self._get_handler(item.source_id)
        dataset = await handler.get_dataset(
            entity_id=item.id_,
            title=item.title,
            config=item.details,
            auth_context=auth_context,
            allow_offline=allow_offline,
        )

        return DataSetSerializer.db_to_schema(item, dataset, expand=True)

    async def get_channel_datasets_count(
        self, channel_id: int | None = None, dataset_id: int | None = None
    ) -> int:
        query = select(func.count("*")).select_from(models.ChannelDataset)
        if channel_id is not None:
            query = query.where(models.ChannelDataset.channel_id == channel_id)
        if dataset_id is not None:
            query = query.where(models.ChannelDataset.dataset_id == dataset_id)

        async with self._lock_session() as session:
            return (await session.execute(query)).scalar_one()

    async def get_channel_dataset_models(
        self, limit: int | None, offset: int, channel_id: int
    ) -> list[models.ChannelDataset]:
        query = select(models.ChannelDataset).where(models.ChannelDataset.channel_id == channel_id)
        async with self._lock_session() as session:
            q_result = await session.scalars(query.limit(limit).offset(offset))
        return [item for item in q_result.all()]

    async def get_channel_dataset_models_with_ds(
        self, channel_id: int
    ) -> list[models.ChannelDataset]:
        query = (
            select(models.ChannelDataset)
            .where(models.ChannelDataset.channel_id == channel_id)
            .options(joinedload(models.ChannelDataset.dataset))
        )

        async with self._lock_session() as session:
            q_result = await session.scalars(query)
        return [item for item in q_result.all()]

    async def get_channel_dataset_schemas(
        self, limit: int | None, offset: int, channel_id: int, auth_context: AuthContext
    ) -> list[schemas.ChannelDatasetExpanded]:
        items = await self.get_channel_dataset_models(
            limit=limit, offset=offset, channel_id=channel_id
        )

        datasets_ids = {d.dataset_id for d in items}
        datasets = await self.get_datasets_schemas(
            limit=None, offset=0, ids=datasets_ids, auth_context=auth_context, allow_offline=True
        )

        channel_dataset_ids = [item.id for item in items]
        latest_versions = await self._get_latest_channel_dataset_versions(channel_dataset_ids)
        latest_successful_versions = await self._get_latest_successful_channel_dataset_versions(
            channel_dataset_ids
        )

        res = []
        for item in items:
            latest_version = latest_versions.get(item.id)
            ds_completed_versions = latest_successful_versions[item.id]
            dataset = next(d for d in datasets if d.id == item.dataset_id)
            res.append(
                ChannelDataSetSerializer.db_to_schema(
                    item, dataset, latest_version, ds_completed_versions
                )
            )
        return res

    async def _get_channel_dataset_model_or_raise(self, item_id: int) -> models.ChannelDataset:
        item: models.ChannelDataset | None = await self._session.get(models.ChannelDataset, item_id)
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Channel dataset not found"
            )
        return item

    async def get_channel_dataset_model_or_none(
        self, channel_id: int, dataset_id: int
    ) -> models.ChannelDataset | None:
        query = (
            select(models.ChannelDataset)
            .where(models.ChannelDataset.channel_id == channel_id)
            .where(models.ChannelDataset.dataset_id == dataset_id)
        )
        async with self._lock_session() as session:
            q_result = await session.scalars(query)
        items = q_result.all()

        if not items:
            return None
        return items[0]

    async def get_channel_dataset_model_or_raise(
        self, channel_id: int, dataset_id: int
    ) -> models.ChannelDataset:
        item = await self.get_channel_dataset_model_or_none(channel_id, dataset_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Channel dataset not found"
            )
        return item

    async def get_channel_dataset_schema(
        self, channel_id: int, dataset_id: int, auth_context: AuthContext
    ) -> schemas.ChannelDatasetExpanded:
        item = await self.get_channel_dataset_model_or_raise(
            channel_id=channel_id, dataset_id=dataset_id
        )
        dataset = await self.get_schema_by_id(item.dataset_id, auth_context, allow_offline=True)
        latest_version = await self._get_latest_channel_dataset_version_schema(item.id)
        last_completed_versions = await self._get_latest_successful_channel_dataset_versions(
            channel_dataset_ids=[item.id]
        )
        return ChannelDataSetSerializer.db_to_schema(
            item, dataset, latest_version, last_completed_versions[item.id]
        )

    async def _get_channel_dataset_version_or_raise(
        self, item_id: int
    ) -> models.ChannelDatasetVersion:
        item: models.ChannelDatasetVersion | None = await self._session.get(
            models.ChannelDatasetVersion, item_id
        )
        if item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Channel dataset version not found"
            )
        return item

    async def get_channel_dataset_version_models(
        self, limit: int | None, offset: int, channel_dataset_id: int
    ) -> list[models.ChannelDatasetVersion]:
        query = (
            select(models.ChannelDatasetVersion)
            .where(models.ChannelDatasetVersion.channel_dataset_id == channel_dataset_id)
            .order_by(models.ChannelDatasetVersion.version.desc())
            .limit(limit)
            .offset(offset)
        )
        q_result = await self._session.scalars(query)
        return [item for item in q_result.all()]

    async def _get_latest_channel_dataset_version_model(
        self, channel_dataset_id: int
    ) -> models.ChannelDatasetVersion | None:
        query = (
            select(models.ChannelDatasetVersion)
            .where(models.ChannelDatasetVersion.channel_dataset_id == channel_dataset_id)
            .order_by(models.ChannelDatasetVersion.version.desc())
            .limit(1)
        )
        q_result = await self._session.scalars(query)
        return q_result.first()

    async def _get_latest_channel_dataset_version_schema(
        self, channel_dataset_id: int
    ) -> schemas.ChannelDatasetVersion | None:
        model = await self._get_latest_channel_dataset_version_model(channel_dataset_id)
        if model is None:
            return None
        return schemas.ChannelDatasetVersion.model_validate(model, from_attributes=True)

    async def _get_latest_channel_dataset_versions(
        self, channel_dataset_ids: list[int]
    ) -> dict[int, schemas.ChannelDatasetVersion]:
        """Get the most recent version regardless of status for each channel dataset."""
        if not channel_dataset_ids:
            return {}

        # Subquery to rank versions by descending order for each channel dataset
        ranked_versions = (
            select(
                models.ChannelDatasetVersion,
                func.row_number()
                .over(
                    partition_by=models.ChannelDatasetVersion.channel_dataset_id,
                    order_by=models.ChannelDatasetVersion.version.desc(),
                )
                .label("rank"),
            )
            .where(models.ChannelDatasetVersion.channel_dataset_id.in_(channel_dataset_ids))
            .subquery()
        )

        # Select only the first row (latest version) for each channel dataset
        query = select(ranked_versions).where(ranked_versions.c.rank == 1)

        result = await self._session.execute(query)
        versions = result.fetchall()

        result_dict = {}
        for row in versions:
            version = schemas.ChannelDatasetVersion.model_validate(row, from_attributes=True)
            result_dict[version.channel_dataset_id] = version

        return result_dict

    async def _get_latest_successful_channel_dataset_versions(
        self, channel_dataset_ids: list[int]
    ) -> dict[int, LastCompletedVersions]:
        """Get the last two versions with the status "COMPLETED" for each channel dataset.
        The result is guaranteed to contain an entry for each channel dataset ID in the input,
        even if no successful version exists (in which case the value will be a tuple of Nones).

        Args:
            channel_dataset_ids: List of channel dataset IDs to filter by.

        Returns:
            A dictionary mapping channel dataset IDs to tuple of their latest successful version
            and the previous successful version. Returns `None` if no suitable version exists.
        """
        if not channel_dataset_ids:
            return {}

        # Subquery to rank successful versions by descending order for each channel dataset
        ranked_versions = (
            select(
                models.ChannelDatasetVersion,
                func.row_number()
                .over(
                    partition_by=models.ChannelDatasetVersion.channel_dataset_id,
                    order_by=models.ChannelDatasetVersion.version.desc(),
                )
                .label("rank"),
            )
            .where(models.ChannelDatasetVersion.channel_dataset_id.in_(channel_dataset_ids))
            .where(models.ChannelDatasetVersion.preprocessing_status == StatusEnum.COMPLETED)
            .subquery()
        )

        # Select the first two rows (latest and previous successful versions) for each channel dataset
        query = select(ranked_versions).where(ranked_versions.c.rank.in_([1, 2]))

        result = await self._session.execute(query)
        versions = result.fetchall()

        result_dict = {cd_id: LastCompletedVersions(None, None) for cd_id in channel_dataset_ids}
        for row in versions:
            version = schemas.ChannelDatasetVersion.model_validate(row, from_attributes=True)
            current = result_dict[version.channel_dataset_id]
            if row.rank == 1:
                result_dict[version.channel_dataset_id] = LastCompletedVersions(
                    version, current.previous_completed_version
                )
            elif row.rank == 2:
                result_dict[version.channel_dataset_id] = LastCompletedVersions(
                    current.last_completed_version, version
                )
            else:
                _log.warning(f"Unexpected rank {row.rank} for version {version.id}: {row}")
        return result_dict

    async def _get_latest_successful_dataset_version(
        self, channel_dataset_ids: list[int]
    ) -> defaultdict[int, LastCompletedVersions]:
        """Get the last two versions with the status "COMPLETED" for each dataset.

        Args:
            channel_dataset_ids: List of channel dataset IDs to filter by.

        Returns:
            A dictionary mapping dataset IDs to the tuple of their latest successful version
            and the previous successful version. Returns `None` if no suitable version exists.
        """
        result_dict: defaultdict[int, LastCompletedVersions] = defaultdict(
            lambda: LastCompletedVersions(None, None)
        )

        if not channel_dataset_ids:
            return result_dict

        # Subquery to rank successful versions by descending order for each channel dataset
        ranked_versions = (
            select(
                models.ChannelDatasetVersion,
                models.ChannelDataset.dataset_id,
                (
                    func.row_number()
                    .over(
                        partition_by=models.ChannelDatasetVersion.channel_dataset_id,
                        order_by=models.ChannelDatasetVersion.version.desc(),
                    )
                    .label("rank")
                ),
            )
            .join(models.ChannelDataset)
            .where(models.ChannelDatasetVersion.channel_dataset_id.in_(channel_dataset_ids))
            .where(models.ChannelDatasetVersion.preprocessing_status == StatusEnum.COMPLETED)
            .subquery()
        )

        # Select the first two rows (latest and previous successful versions) for each channel dataset
        query = select(ranked_versions).where(ranked_versions.c.rank.in_([1, 2]))

        result = await self._session.execute(query)
        versions = result.fetchall()

        for row in versions:
            version = schemas.ChannelDatasetVersion.model_validate(row, from_attributes=True)
            current = result_dict[row.dataset_id]
            if row.rank == 1:
                result_dict[row.dataset_id] = LastCompletedVersions(
                    version, current.previous_completed_version
                )
            elif row.rank == 2:
                result_dict[row.dataset_id] = LastCompletedVersions(
                    current.last_completed_version, version
                )
            else:
                _log.warning(f"Unexpected rank {row.rank} for version {version.id}: {row}")
        return result_dict

    async def get_channel_dataset_versions_schemas(
        self, limit: int | None, offset: int, channel_id: int, dataset_id: int
    ) -> list[schemas.ChannelDatasetVersion]:
        channel_dataset = await self.get_channel_dataset_model_or_raise(channel_id, dataset_id)
        items = await self.get_channel_dataset_version_models(
            limit=limit, offset=offset, channel_dataset_id=channel_dataset.id
        )
        return [
            schemas.ChannelDatasetVersion.model_validate(item, from_attributes=True)
            for item in items
        ]

    async def get_channel_dataset_versions_count(self, channel_id: int, dataset_id: int) -> int:
        channel_dataset = await self.get_channel_dataset_model_or_raise(channel_id, dataset_id)
        query = (
            select(func.count("*"))
            .select_from(models.ChannelDatasetVersion)
            .where(models.ChannelDatasetVersion.channel_dataset_id == channel_dataset.id)
        )
        return (await self._session.execute(query)).scalar_one()

    async def get_latest_successful_dataset_versions_for_channel(
        self, channel_id: int
    ) -> defaultdict[int, LastCompletedVersions]:
        """Get the latest successful dataset versions for all datasets in a channel.

        Args:
            channel_id: The ID of the channel to filter datasets by.
        Returns:
            A dictionary mapping dataset IDs to the tuple of their latest successful version
            and the previous successful version. Returns `None` if no suitable version exists.
        """

        channel_datasets = await self.get_channel_dataset_models_with_ds(channel_id)
        channel_dataset_ids = [cd.id for cd in channel_datasets]
        return await self._get_latest_successful_dataset_version(channel_dataset_ids)

import asyncio
from collections.abc import Iterable

from fastapi import HTTPException, status
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload
from sqlalchemy.sql.expression import func

import common.models as models
import common.schemas as schemas
from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.base import DataSet, DataSourceHandler

from .data_source import DataSourceSerializer, DataSourceService, DataSourceTypeService


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
        item_db: models.ChannelDataset, dataset: schemas.DataSet
    ) -> schemas.ChannelDatasetExpanded:
        return schemas.ChannelDatasetExpanded(
            id=item_db.id,
            created_at=item_db.created_at,
            updated_at=item_db.updated_at,
            channel_id=item_db.channel_id,
            dataset_id=item_db.dataset_id,
            preprocessing_status=item_db.preprocessing_status,
            dataset=dataset,
        )


class DataSetService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        return (await self._session.execute(query)).scalar_one()

    async def get_datasets_models(
        self,
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

        q_result = await self._session.scalars(query.limit(limit).offset(offset))
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
                    entity_id=str(item.id),
                    title=item.title,
                    config=item.details,
                    auth_context=auth_context,
                    allow_offline=allow_offline,
                )
            )

        datasets: list[DataSet] = await asyncio.gather(*tasks)

        return [
            DataSetSerializer.db_to_schema(item, ds, expand=True)
            for item, ds in zip(items, datasets)
        ]

    async def _get_handler(self, data_source_id: int) -> DataSourceHandler:
        source_service = DataSourceService(self._session)
        source_type_service = DataSourceTypeService(self._session)

        source: models.DataSource = await source_service.get_by_id(data_source_id)
        handler_class = await source_type_service.get_data_source_handler_class_by_id(
            source.type_id
        )

        config = handler_class.parse_config(source.details)
        logger.info(f"{config=}")

        return handler_class(config=config)

    async def _get_item_or_raise(self, item_id: int, expand: bool = False) -> models.DataSet:
        options = None
        if expand:
            options = [selectinload(models.DataSet.source).selectinload(models.DataSource.type)]
        item: models.DataSet | None = await self._session.get(
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
        await self._session.refresh(item, attribute_names=["source"])
        return item

    async def get_schema_by_id(
        self, item_id: int, auth_context: AuthContext, allow_offline: bool = False
    ) -> schemas.DataSet:
        item = await self.get_model_by_id(item_id, expand=True)

        handler = await self._get_handler(item.source_id)
        dataset = await handler.get_dataset(
            entity_id=str(item.id),
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

        return (await self._session.execute(query)).scalar_one()

    async def get_channel_dataset_models(
        self, limit: int | None, offset: int, channel_id: int
    ) -> list[models.ChannelDataset]:
        query = select(models.ChannelDataset).where(models.ChannelDataset.channel_id == channel_id)
        q_result = await self._session.scalars(query.limit(limit).offset(offset))
        return [item for item in q_result.all()]

    async def get_channel_dataset_models_with_ds(
        self, limit: int | None, offset: int, channel_id: int
    ) -> list[models.ChannelDataset]:
        query = (
            select(models.ChannelDataset)
            .where(models.ChannelDataset.channel_id == channel_id)
            .options(joinedload(models.ChannelDataset.dataset))
        )
        q_result = await self._session.scalars(query.limit(limit).offset(offset))
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

        res = []
        for item in items:
            dataset = next(d for d in datasets if d.id == item.dataset_id)
            res.append(ChannelDataSetSerializer.db_to_schema(item, dataset))
        return res

    async def get_channel_dataset_model_or_none(
        self, channel_id: int, dataset_id: int
    ) -> models.ChannelDataset | None:
        query = (
            select(models.ChannelDataset)
            .where(models.ChannelDataset.channel_id == channel_id)
            .where(models.ChannelDataset.dataset_id == dataset_id)
        )
        q_result = await self._session.scalars(query)
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
        self, channel_id: int, dataset_id: int
    ) -> schemas.ChannelDatasetBase:
        item = await self.get_channel_dataset_model_or_raise(
            channel_id=channel_id, dataset_id=dataset_id
        )
        return schemas.ChannelDatasetBase.model_validate(item, from_attributes=True)

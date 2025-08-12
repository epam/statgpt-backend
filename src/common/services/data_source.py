from typing import Iterable

from fastapi import HTTPException, status
from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql.expression import func

import common.models as models
import common.schemas as schemas
from common.config import multiline_logger as logger
from common.data import DataManager, DataSourceConfig, DataSourceHandler


class DataSourceTypeSerializer:
    @staticmethod
    def db_to_schema(item_db: models.DataSourceType) -> schemas.DataSourceType:
        return schemas.DataSourceType(
            id=item_db.id,
            created_at=item_db.created_at,
            updated_at=item_db.updated_at,
            name=item_db.name,
            description=item_db.description,
        )


class DataSourceTypeService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_count(self) -> int:
        query = select(func.count("*")).select_from(models.DataSourceType)  # type: ignore
        return (await self._session.execute(query)).scalar_one()

    async def get_data_source_types(self, limit: int, offset: int) -> list[schemas.DataSourceType]:
        query = select(models.DataSourceType).limit(limit).offset(offset)

        q_result = await self._session.execute(query)

        return [DataSourceTypeSerializer.db_to_schema(item) for item in q_result.scalars().all()]

    async def _get_item_or_raise(self, item_id: int) -> models.DataSourceType:
        item: models.DataSourceType | None = await self._session.get(models.DataSourceType, item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DataSourceType with id={item_id} not found",
            )
        return item

    async def get_by_id(self, item_id: int) -> models.DataSourceType:
        return await self._get_item_or_raise(item_id)

    async def get_config_class(self, source_type_id: int) -> type[DataSourceConfig]:
        item = await self._get_item_or_raise(source_type_id)
        logger.info(f"Searching config for {item}")

        config_class = DataManager.get_config_class(item.name)
        logger.info(f"{config_class=}")
        return config_class

    async def get_data_source_handler_class_by_id(self, item_id: int) -> type[DataSourceHandler]:
        item = await self._get_item_or_raise(item_id)
        logger.info(f"Searching handler class for {item}")
        return await self.get_data_source_handler_class(item)

    @staticmethod
    async def get_data_source_handler_class(item: models.DataSourceType) -> type[DataSourceHandler]:
        handler_class = DataManager.get_data_source_handler_class(item.name)
        logger.info(f"{handler_class=}")
        return handler_class

    async def get_schema_config(self, item_id: int) -> dict:
        config_class = await self.get_config_class(item_id)
        return config_class.model_json_schema()


class DataSourceSerializer:
    @staticmethod
    def db_to_schema(item_db: models.DataSource) -> schemas.DataSource:
        return schemas.DataSource(
            id=item_db.id,
            created_at=item_db.created_at,
            updated_at=item_db.updated_at,
            title=item_db.title,
            description=item_db.description,
            type_id=item_db.type_id,
            details=item_db.details,
            type=DataSourceTypeSerializer.db_to_schema(item_db.type),
        )


class DataSourceService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def _apply_filters(query: Select, ids: Iterable[int] | None) -> Select:

        if ids is not None:
            query = query.where(models.DataSource.id.in_(ids))

        return query

    async def get_data_sources_count(self, ids: Iterable[int] | None = None) -> int:
        query = select(func.count("*")).select_from(models.DataSource)  # type: ignore
        query = self._apply_filters(query, ids)
        return (await self._session.execute(query)).scalar_one()

    async def get_data_sources_models(
        self, limit: int | None, offset: int, ids: Iterable[int] | None = None
    ) -> list[models.DataSource]:

        query = select(models.DataSource).options(selectinload(models.DataSource.type))
        query = self._apply_filters(query, ids)

        q_result = await self._session.execute(query.limit(limit).offset(offset))
        return [item for item in q_result.scalars().all()]

    async def get_data_sources_schemas(
        self,
        limit: int | None,
        offset: int,
        ids: Iterable[int] | None = None,
    ) -> list[schemas.DataSource]:
        items = await self.get_data_sources_models(limit=limit, offset=offset, ids=ids)
        return [DataSourceSerializer.db_to_schema(item) for item in items]

    async def _get_item_or_raise(self, item_id: int) -> models.DataSource:
        item: models.DataSource | None = await self._session.get(models.DataSource, item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DataSource with id={item_id} not found",
            )
        return item

    async def get_by_id(self, item_id: int) -> models.DataSource:
        item = await self._get_item_or_raise(item_id)
        await self._session.refresh(item, attribute_names=["type"])
        return item

    async def get_schema_by_id(self, item_id: int) -> schemas.DataSource:
        item = await self.get_by_id(item_id)
        return DataSourceSerializer.db_to_schema(item)

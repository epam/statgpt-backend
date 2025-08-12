from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import func

import common.models as models
import common.schemas as schemas


class ChannelSerializer:
    @staticmethod
    def db_to_schema(channel_db: models.Channel) -> schemas.Channel:
        return schemas.Channel.model_validate(channel_db, from_attributes=True)


class ChannelService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_channels_count(self) -> int:
        query = select(func.count("*")).select_from(models.Channel)  # type: ignore
        return (await self._session.execute(query)).scalar_one()

    async def get_channels_db(self, limit: int | None, offset: int) -> list[models.Channel]:
        query = select(models.Channel).limit(limit).offset(offset)

        q_result = await self._session.execute(query)
        return [item for item in q_result.scalars().all()]

    async def get_channels_schemas(self, limit: int, offset: int) -> list[schemas.Channel]:
        channels = await self.get_channels_db(limit, offset)
        return [ChannelSerializer.db_to_schema(item) for item in channels]

    async def get_channel_by_deployment_id(self, deployment_id: str) -> models.Channel:
        query = select(models.Channel).where(models.Channel.deployment_id == deployment_id)
        q_result = await self._session.execute(query)
        return q_result.scalar_one()

    async def _get_item_or_raise(self, item_id: int) -> models.Channel:
        item: models.Channel | None = await self._session.get(models.Channel, item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Channel with id={item_id} not found"
            )
        return item

    async def get_model_by_id(self, item_id: int) -> models.Channel:
        return await self._get_item_or_raise(item_id)

    async def get_schema_by_id(self, item_id: int) -> schemas.Channel:
        item = await self.get_model_by_id(item_id)
        return ChannelSerializer.db_to_schema(item)

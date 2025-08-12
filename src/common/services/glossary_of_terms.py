from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from common import models, schemas


class GlossaryOfTermsService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_terms_count(self, channel_id: int) -> int:
        query = (
            select(func.count("*"))
            .select_from(models.GlossaryTerm)
            .where(models.GlossaryTerm.channel_id == channel_id)
        )
        return (await self._session.execute(query)).scalar_one()

    async def get_term_models_by_channel(
        self, channel_id: int, limit: int | None, offset: int
    ) -> list[models.GlossaryTerm]:
        query = (
            select(models.GlossaryTerm)
            .where(models.GlossaryTerm.channel_id == channel_id)
            .limit(limit)
            .offset(offset)
        )

        q_result = await self._session.execute(query)
        return [item for item in q_result.scalars().all()]

    async def _get_term_models_by_ids(self, term_ids: list[int]) -> list[models.GlossaryTerm]:
        query = select(models.GlossaryTerm).where(models.GlossaryTerm.id.in_(term_ids))
        q_result = await self._session.execute(query)
        return [item for item in q_result.scalars().all()]

    async def get_term_schemas_by_channel(
        self, channel_id: int, limit: int, offset: int
    ) -> list[schemas.GlossaryTerm]:
        terms = await self.get_term_models_by_channel(channel_id, limit, offset)
        return [schemas.GlossaryTerm.model_validate(item, from_attributes=True) for item in terms]

    async def _get_item_or_raise(self, item_id: int) -> models.GlossaryTerm:
        item: models.GlossaryTerm | None = await self._session.get(models.GlossaryTerm, item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"GlossaryTerm with id={item_id} not found",
            )
        return item

    async def get_term_schema_by_id(self, item_id: int) -> schemas.GlossaryTerm:
        item = await self._get_item_or_raise(item_id)
        return schemas.GlossaryTerm.model_validate(item, from_attributes=True)

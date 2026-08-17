import asyncio

from sqlalchemy import ColumnElement, Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.common import models, schemas
from statgpt.common.services.base import DbServiceBase
from statgpt.common.utils import normalize_whitespace

RecordKey = tuple[str, str]
"""The natural key of a record within its channel: folded (agency, dataset_id)."""


def normalize_key_part(value: str) -> str:
    """Fold one half of the natural key the way the database's generated column does.

    ``.lower()``, never ``.casefold()``: the generated `agency_key` / `dataset_id_key`
    columns hold what PostgreSQL's ``lower()`` produced, and ``casefold()`` is more
    aggressive (``'ß'`` -> ``'ss'``), so it would miss rows the database considers matches.

    Whitespace is normalized first, because that is what the write path stored: a value
    only stripped at the ends would not match a stored one whose internal runs were
    collapsed.
    """
    return normalize_whitespace(value).lower()


def record_key(agency: str, dataset_id: str) -> RecordKey:
    """Build the key the database compares on."""
    return normalize_key_part(agency), normalize_key_part(dataset_id)


class DiscoveryDatasetService(DbServiceBase):
    """Read access to the discovery dataset records of a channel.

    Lives in `common` because the read side is shared: the chat application needs the
    same records the admin portal writes, and `common` cannot import from `admin`. Nothing
    here raises on a missing record - the caller decides what an absent id means.
    """

    def __init__(self, session: AsyncSession, session_lock: asyncio.Lock | None = None) -> None:
        super().__init__(session, session_lock)

    @staticmethod
    def _filters(
        channel_id: int,
        validation_status: schemas.DiscoveryValidationStatus | None = None,
        indexing_status: schemas.DiscoveryIndexingStatus | None = None,
        agency: str | None = None,
    ) -> list[ColumnElement[bool]]:
        clauses: list[ColumnElement[bool]] = [models.DiscoveryDataset.channel_id == channel_id]
        if validation_status is not None:
            clauses.append(models.DiscoveryDataset.validation_status == validation_status)
        if indexing_status is not None:
            clauses.append(models.DiscoveryDataset.indexing_status == indexing_status)
        if agency is not None:
            # Matched through the generated key, and folded by the same helper the write
            # path uses, so the filter is case- and whitespace-insensitive in exactly the
            # way the natural key is.
            clauses.append(models.DiscoveryDataset.agency_key == normalize_key_part(agency))
        return clauses

    def _select_by_channel(
        self,
        channel_id: int,
        validation_status: schemas.DiscoveryValidationStatus | None = None,
        indexing_status: schemas.DiscoveryIndexingStatus | None = None,
        agency: str | None = None,
    ) -> Select[tuple[models.DiscoveryDataset]]:
        return select(models.DiscoveryDataset).where(
            *self._filters(channel_id, validation_status, indexing_status, agency)
        )

    async def get_records_count(
        self,
        channel_id: int,
        validation_status: schemas.DiscoveryValidationStatus | None = None,
        indexing_status: schemas.DiscoveryIndexingStatus | None = None,
        agency: str | None = None,
    ) -> int:
        query = (
            select(func.count("*"))
            .select_from(models.DiscoveryDataset)
            .where(*self._filters(channel_id, validation_status, indexing_status, agency))
        )
        async with self._lock_session() as session:
            return (await session.execute(query)).scalar_one()

    async def get_record_models_by_channel(
        self,
        channel_id: int,
        limit: int | None,
        offset: int,
        validation_status: schemas.DiscoveryValidationStatus | None = None,
        indexing_status: schemas.DiscoveryIndexingStatus | None = None,
        agency: str | None = None,
    ) -> list[models.DiscoveryDataset]:
        query = (
            self._select_by_channel(channel_id, validation_status, indexing_status, agency)
            .order_by(models.DiscoveryDataset.id)
            .limit(limit)
            .offset(offset)
        )
        async with self._lock_session() as session:
            q_result = await session.execute(query)
        return list(q_result.scalars().all())

    async def get_record_schemas_by_channel(
        self,
        channel_id: int,
        limit: int | None,
        offset: int,
        validation_status: schemas.DiscoveryValidationStatus | None = None,
        indexing_status: schemas.DiscoveryIndexingStatus | None = None,
        agency: str | None = None,
    ) -> list[schemas.DiscoveryDataset]:
        records = await self.get_record_models_by_channel(
            channel_id, limit, offset, validation_status, indexing_status, agency
        )
        return [self._serialize(item) for item in records]

    async def get_record_models_by_ids(self, item_ids: list[int]) -> list[models.DiscoveryDataset]:
        query = select(models.DiscoveryDataset).where(models.DiscoveryDataset.id.in_(item_ids))
        async with self._lock_session() as session:
            q_result = await session.execute(query)
        return list(q_result.scalars().all())

    async def get_record_model_by_id(self, item_id: int) -> models.DiscoveryDataset | None:
        async with self._lock_session() as session:
            return await session.get(models.DiscoveryDataset, item_id)

    @staticmethod
    def _serialize(item: models.DiscoveryDataset) -> schemas.DiscoveryDataset:
        return schemas.DiscoveryDataset.model_validate(item, from_attributes=True)

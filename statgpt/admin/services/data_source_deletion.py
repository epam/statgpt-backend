from sqlalchemy.ext.asyncio import AsyncSession

from .data_source import AdminPortalDataSourceService
from .dataset import AdminPortalDataSetService


class AdminPortalDataSourceDeletionService:
    """Deletes datasource with audited deletion of related datasets."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._data_source_service = AdminPortalDataSourceService(session)
        self._dataset_service = AdminPortalDataSetService(session)

    async def delete(self, item_id: int) -> None:
        # Keep dataset deletions and datasource deletion in one transaction,
        # while still using audited service-level delete flows.
        async with self._session.begin():
            datasets = await self._dataset_service.get_datasets_models(
                limit=None,
                offset=0,
                source_id=item_id,
            )
            for dataset in datasets:
                await self._dataset_service.delete(dataset.id)

            await self._data_source_service.delete(item_id)

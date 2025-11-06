import typing

import sdmx
from sdmx.message import StructureMessage

from common.data.base.dataset_hierarchy import (
    DatasetHierarchy,
    DatasetHierarchyCreatorABC,
    HierarchyItem,
)
from common.data.sdmx.common.config import CategorySchemaDataSetHierarchyConfig

if typing.TYPE_CHECKING:
    from common.data.sdmx import Sdmx21DataSourceHandler


class CategorySchemaDataSetHierarchyCreator(DatasetHierarchyCreatorABC):
    """Loads SDMX category scheme and creates dataset hierarchy from it."""

    _handler: 'Sdmx21DataSourceHandler'

    async def create_hierarchy(self) -> DatasetHierarchy:
        struct_msg = await self._load_hierarchy()
        categories = self._get_categories_from(struct_msg)
        datasets = self._get_datasets_from(struct_msg)
        return DatasetHierarchy(categories + datasets)

    async def _load_hierarchy(self) -> StructureMessage:
        client = await self._handler.create_sdmx_client(self._auth_context)
        return await client.categoryscheme(
            agency_id=self._config.agency_id,
            resource_id=self._config.resource_id,
            version=self._config.version,
        )

    @property
    def _config(self) -> CategorySchemaDataSetHierarchyConfig:
        return self._handler.config.dataset_hierarchy_config  # type: ignore

    @property
    def _locale(self) -> str:
        return self._handler.config.locale

    def _get_categories_from(self, struct_msg: StructureMessage) -> list[HierarchyItem]:
        parsed_response = sdmx.to_pandas(struct_msg, locale=self._locale)
        if 'category_scheme' not in parsed_response:
            raise ValueError("Category scheme not found in the structure message")

        df = parsed_response['category_scheme'][self._config.resource_id]
        return [
            HierarchyItem(
                entity_id=entity_id, name=name, parent_entity_id=parent_id or None, is_dataset=False
            )
            for entity_id, name, parent_id in df.itertuples(index=True)
        ]

    def _get_datasets_from(self, struct_msg: StructureMessage) -> list[HierarchyItem]:
        res = []
        for item in struct_msg.categorisation.values():
            dataflow_urn = (
                f"{item.artefact.maintainer.id}:{item.artefact.id}({item.artefact.version})"
            )
            res.append(
                HierarchyItem(
                    entity_id=dataflow_urn,
                    name=item.artefact.name[self._locale],
                    parent_entity_id=item.category.id,
                    is_dataset=True,
                )
            )
        return res

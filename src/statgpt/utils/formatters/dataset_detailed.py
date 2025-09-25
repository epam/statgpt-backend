import random
from collections.abc import Sequence

from common.data.base import CategoricalDimension, Category, DataSet

from .dataset_simple import SimpleDatasetFormatter


class DetailedDatasetFormatter(SimpleDatasetFormatter):

    @staticmethod
    def _format_values(values: Sequence[Category]) -> str:
        return ", ".join([f"{v.name} [{v.query_id}]" for v in values])

    def _format_component_values(
        self, values: Sequence[Category], limit: int, shuffle_sample: bool
    ) -> list[str]:
        if len(values) <= limit:
            values_str = self._format_values(values)
            return [
                f'{self._("Total")}: {len(values)} {self._("items")}',
                f'{self._("Values")}: {values_str}',
            ]
        sample_values: Sequence[Category]
        if shuffle_sample:
            sample_values = random.sample(values, limit)
        else:
            sample_values = values[:limit]
        sample_values_str = self._format_values(sample_values)
        return [
            f'{self._("Total")}: {len(values)} {self._("items")}',
            f'{self._("Sample values")}: {sample_values_str}...',
        ]

    def _append_dimensions(self, dataset: DataSet, result: list[str]):
        dimensions = dataset.dimensions()
        if dimensions:
            base_tabs = '\t' * self.config.list_level
            result.append(f'{base_tabs}- {self._("Dimensions")} ({len(dimensions)})')

            for dim in dimensions:
                dim_name = dim.name if hasattr(dim, 'name') else str(dim)
                dim_id = dim.entity_id if hasattr(dim, 'entity_id') else ''
                dimension_tabs = '\t' * (self.config.list_level + 1)

                if dim_id:
                    result.append(
                        f'{dimension_tabs}- **{dim_name}** [{dim_id}] - {self._("Type")}: {dim.dimension_type}'
                    )
                else:
                    result.append(
                        f'{dimension_tabs}- **{dim_name}** - {self._("Type")}: {dim.dimension_type}'
                    )

                dimension_details_tabs = '\t' * (self.config.list_level + 2)

                if dim.description:
                    result.append(
                        f'{dimension_details_tabs}- {self._("Description")}: {dim.description}'
                    )

                if isinstance(dim, CategoricalDimension):
                    values = dim.available_values
                    formatted_values = self._format_component_values(
                        values, limit=10, shuffle_sample=True
                    )
                    for value in formatted_values:
                        result.append(f'{dimension_details_tabs}- {value}')

    def _append_attributes(self, dataset: DataSet, result: list[str]) -> None:
        attributes = dataset.attributes()
        if attributes:
            base_tabs = '\t' * self.config.list_level
            result.append(f'{base_tabs}- {self._("Attributes")} ({len(attributes)})')

            for attr in attributes:
                attr_name = attr.name if hasattr(attr, 'name') else str(attr)
                attr_id = attr.entity_id if hasattr(attr, 'entity_id') else ''

                attribute_tabs = '\t' * (self.config.list_level + 1)
                if attr_id:
                    result.append(
                        f'{attribute_tabs}- **{attr_name}** [{attr_id}] - {self._("Type")}: {attr.attribute_type}'
                    )
                else:
                    result.append(
                        f'{attribute_tabs}- **{attr_name}** - {self._("Type")}: {attr.attribute_type}'
                    )

                attribute_details_tabs = '\t' * (self.config.list_level + 2)
                if attr.description:
                    result.append(
                        f'{attribute_details_tabs}- {self._("Description")}: {attr.description}'
                    )

    async def format(self, dataset: DataSet) -> str:
        result: list[str] = []
        await self._append_basic_info(dataset, result)
        self._append_dimensions(dataset, result)
        self._append_attributes(dataset, result)
        return "\n".join(result)

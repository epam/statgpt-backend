from collections.abc import Sequence

from common.data.base import CategoricalDimension, Category, DataSet

from .citation import CitationOverrideFormatter
from .dataset_base import BaseDatasetFormatter


class DetailedDatasetFormatter(BaseDatasetFormatter):

    @staticmethod
    def _format_values(values: Sequence[Category]) -> str:
        return ", ".join([f"{v.name} [{v.query_id}]" for v in values])

    def _append_dimensions(self, dataset: DataSet, result: list[str]):
        dimensions = dataset.dimensions()
        if dimensions:
            result.append(f'\n#### {self._("Dimensions")} ({len(dimensions)})')

            for dim in dimensions:
                dim_name = dim.name if hasattr(dim, 'name') else str(dim)
                dim_id = dim.entity_id if hasattr(dim, 'entity_id') else ''

                if dim_id:
                    result.append(f'- **{dim_name}** [{dim_id}]')
                else:
                    result.append(f'- **{dim_name}**')

                result.append(f'  - {self._("Type")}: {dim.dimension_type}')

                # Add dimension details if available

                if dim.description:
                    result.append(f'  - {self._("Description")}: {dim.description}')

                if isinstance(dim, CategoricalDimension):
                    values = dim.available_values
                    if values and len(values) <= 10:
                        values_str = self._format_values(values)
                        result.append(f'  - {self._("Values")}: {values_str}')
                    elif values:
                        sample_values = self._format_values(values[:10])
                        result.append(
                            f'  - {self._("Total")}: {len(values)} {self._("items")}, {self._("Sample values")}: {sample_values}...'
                        )

    def _append_attributes(self, dataset: DataSet, result: list[str]):
        attributes = dataset.attributes()
        if attributes:
            result.append(f'\n#### {self._("Attributes")} ({len(attributes)})')

            for attr in attributes:
                attr_name = attr.name if hasattr(attr, 'name') else str(attr)
                attr_id = attr.entity_id if hasattr(attr, 'entity_id') else ''

                if attr_id:
                    result.append(f'- **{attr_name}** [{attr_id}]')
                else:
                    result.append(f'- **{attr_name}**')

                result.append(f'  - {self._("Type")}: {attr.attribute_type}')

                if attr.description:
                    result.append(f'  - {self._("Description")}: {attr.description}')

    async def format(self, dataset: DataSet) -> str:
        result = []

        # Dataset name
        if self.config.include_name:
            name_str = f'**{dataset.name}**' if self.config.highlight_name_in_bold else dataset.name
            if self.config.official_dataset_label and dataset.config.is_official:
                official_label = (
                    self._(self.config.official_dataset_label)
                    if self.config.official_dataset_label
                    else self._("[Official]")
                )
                name_str += f' {official_label}'
            result.append(f'### {name_str}')

        # Basic Information section
        result.append(f'\n#### {self._("Basic Information")}')

        if self.config.add_source_id:
            id_label = self.config.source_id_name or self._("ID")
            result.append(f'- **{id_label}**: {dataset.source_id}')

        if self.config.add_entity_id:
            entity_label = self.config.entity_id_name or self._("Internal ID")
            result.append(f'- **{entity_label}**: {dataset.entity_id}')

        if self.config.use_description:
            citation = dataset.config.citation
            description = (
                citation.description if citation and citation.description else dataset.description
            )
            if description:
                result.append(f'- **{self._("Description")}**: {description}')

        citation = dataset.config.citation
        if citation and self.config.citation and self.config.citation.is_use_any:
            last_updated = await self._get_dataset_update_at(dataset)
            formatted_citation = await CitationOverrideFormatter(
                config=self.config.citation,
                locale=self.config.locale,
                last_updated_override_value=last_updated,
            ).format(citation)
            result.append(formatted_citation)

        self._append_dimensions(dataset, result)
        self._append_attributes(dataset, result)

        return "\n".join(result)

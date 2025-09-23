from common.data.base import DataSet

from .citation import CitationOverrideFormatter
from .dataset_base import BaseDatasetFormatter


class SimpleDatasetFormatter(BaseDatasetFormatter):
    async def format(self, dataset: DataSet) -> str:
        citation = dataset.config.citation
        result = []

        if self.config.include_name:
            name_str = f'**{dataset.name}**' if self.config.highlight_name_in_bold else dataset.name
            if self.config.official_dataset_label and dataset.config.is_official:
                official_label = (
                    self._(self.config.official_dataset_label)
                    if self.config.official_dataset_label
                    else self._("[Official]")
                )
                name_str += f' {official_label}'
            result.append(f'* {name_str}')

        if self.config.add_source_id:
            id_label = self.config.source_id_name or self._("ID")
            result.append(f'\t* {id_label}: {dataset.source_id}')

        if self.config.add_entity_id:
            entity_label = self.config.entity_id_name or self._("Internal ID")
            result.append(f'\t* {entity_label}: {dataset.entity_id}')

        if self.config.use_description:
            description = (
                citation.description if citation and citation.description else dataset.description
            )
            if description:
                result.append(f'\t* {self._("Description")}: {description}')

        if citation and self.config.citation and self.config.citation.is_use_any:
            last_updated = await self._get_dataset_update_at(dataset)
            formatted_citation = await CitationOverrideFormatter(
                config=self.config.citation,
                locale=self.config.locale,
                last_updated_override_value=last_updated,
            ).format(citation)
            result.append(formatted_citation)

        return "\n".join(result)

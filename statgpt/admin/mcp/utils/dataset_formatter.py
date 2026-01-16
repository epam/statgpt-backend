# copied(with removing some logic) from statgpt/app/utils/formatters/dataset_detailed.py
import gettext
import random
from abc import ABC
from collections.abc import Sequence
from pathlib import Path

from statgpt.common.data.base import CategoricalDimension, Category
from statgpt.common.data.sdmx.common import SdmxDimension
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.schemas.enums import LocaleEnum


class BaseFormatter(ABC):
    def __init__(self, domain: str, locale: LocaleEnum) -> None:
        locale_dir = Path(__file__).parent / "locales"
        self.translation: gettext.NullTranslations | gettext.GNUTranslations
        try:
            # noinspection PyTypeChecker
            self.translation = gettext.translation(
                domain, localedir=locale_dir, languages=[locale.value], fallback=False
            )
        except FileNotFoundError:
            # noinspection PyTypeChecker
            self.translation = gettext.translation(
                domain, localedir=locale_dir, languages=[LocaleEnum.EN.value], fallback=True
            )

        self._ = self.translation.gettext


class DetailedDatasetFormatter(BaseFormatter):
    """Formatter for creating detailed dataset structure descriptions."""

    def __init__(
        self, include_name: bool, list_level: int, add_source_id: bool, locale: LocaleEnum
    ) -> None:
        super().__init__("dataset_formatter", locale)
        self.include_name = include_name
        self.list_level = list_level
        self.add_source_id = add_source_id

    def _append_basic_info(self, name: str, result: list[str]) -> None:
        if self.include_name:
            name_str = f'## {name}'
            if self.list_level == 0:
                result.append(f'{name_str}\n')
            else:
                tabs = '\t' * (self.list_level - 1)
                result.append(f'{tabs}* {name_str}')

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

    def _append_dimensions(
        self, dimensions: Sequence[SdmxDimension] | None, result: list[str]
    ) -> None:
        if dimensions:
            base_tabs = '\t' * self.list_level
            result.append(f'\n{base_tabs}- {self._("Dimensions")} ({len(dimensions)})')

            for dim in dimensions:
                dim_name = dim.name if hasattr(dim, 'name') else str(dim)
                dim_id = dim.entity_id if hasattr(dim, 'entity_id') else ''
                dimension_tabs = '\t' * (self.list_level + 1)

                if dim_id:
                    result.append(
                        f'{dimension_tabs}- **{dim_name}** [{dim_id}] - {self._("Type")}: {dim.dimension_type}'
                    )
                else:
                    result.append(
                        f'{dimension_tabs}- **{dim_name}** - {self._("Type")}: {dim.dimension_type}'
                    )

                dimension_details_tabs = '\t' * (self.list_level + 2)

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

    def _append_attributes(
        self, attributes: Sequence[Sdmx21Attribute] | None, result: list[str]
    ) -> None:
        if attributes:
            base_tabs = '\t' * self.list_level
            result.append(f'\n{base_tabs}- {self._("Attributes")} ({len(attributes)})')

            for attr in attributes:
                attr_name = attr.name if hasattr(attr, 'name') else str(attr)
                attr_id = attr.entity_id if hasattr(attr, 'entity_id') else ''

                attribute_tabs = '\t' * (self.list_level + 1)
                if attr_id:
                    result.append(
                        f'{attribute_tabs}- **{attr_name}** [{attr_id}] - {self._("Type")}: {attr.attribute_type}'
                    )
                else:
                    result.append(
                        f'{attribute_tabs}- **{attr_name}** - {self._("Type")}: {attr.attribute_type}'
                    )

                attribute_details_tabs = '\t' * (self.list_level + 2)
                if attr.description:
                    result.append(
                        f'{attribute_details_tabs}- {self._("Description")}: {attr.description}'
                    )

    async def format(
        self,
        name: str,
        dimensions: Sequence[SdmxDimension] | None,
        attributes: Sequence[Sdmx21Attribute] | None,
    ) -> str:
        """Format dataset structure into a detailed markdown string."""
        result: list[str] = []
        self._append_basic_info(name, result)
        self._append_dimensions(dimensions, result)
        self._append_attributes(attributes, result)
        return "\n".join(result)

from typing import Any, Literal

from statgpt.common.data.quanthub.sdmx_schemas.v30 import Attribute, QhDataMessage


class AttributesParser:
    """Parses attributes from a QuantHub SDMX Data Message."""

    @classmethod
    def parse(cls, data_message: QhDataMessage) -> dict[str, str | None]:
        """Extracts attributes and their values from the data message."""
        if not data_message.data or len(data_message.data.data_sets) != 1:
            raise ValueError(f"Unexpected data message response: {data_message}")

        result = {}
        for dataset in data_message.data.data_sets:
            structure = data_message.data.structures[dataset.structure]

            for i, attr_values in enumerate(dataset.attributes):
                if structure.attributes is None:
                    raise ValueError(f"Unexpected structure attributes: {structure}")

                attribute = structure.attributes.data_set[i]
                result[attribute.id] = cls._parse_values(attribute, attr_values)

        return result

    @staticmethod
    def _parse_values(attribute: Attribute, values: list[Any] | Literal[0] | None) -> str | None:
        if values == 0 or values == [0]:
            return attribute.values[0] if attribute.values else None
        if not values or all(v is None for v in values):
            return None
        return ', '.join(str(v) for v in values)

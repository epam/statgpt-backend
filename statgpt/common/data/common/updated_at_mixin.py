import logging
from datetime import datetime

from statgpt.common.auth.auth_context import AuthContext

from .property_source import PropertySource, PropertySourceEnum

_log = logging.getLogger(__name__)


class UpdatedAtMixin:
    """
    Mixin providing `updated_at` resolution from configurable property sources.

    Subclasses must:
    - Have `_config` with `updated_at: list[PropertySource]`
    - Override getter methods for the property sources they support:
      - `_get_annotation_value_by_id` for ANNOTATION
      - `_get_attribute_value_by_id` for ATTRIBUTE
      - `_get_citation_value` for CITATION
    - VALUE source is built-in (returns the field as-is)
    """

    def _get_annotation_value_by_id(self, annotation_id: str) -> str | None:
        """Get value from annotation by ID. Override in subclasses that support annotations."""
        return None

    def _get_attribute_value_by_id(self, attribute_id: str) -> str | None:
        """Get value from attribute by ID. Override in subclasses that support attributes."""
        return None

    def _get_citation_value(self, field: str) -> str | None:
        """Get value from citation config by field name. Override in subclasses that support citation."""
        return None

    def _get_property_value_by_source(self, property_source: PropertySource) -> str | None:
        """Get the property value using the specified property source."""
        mapping = {
            PropertySourceEnum.ANNOTATION: self._get_annotation_value_by_id,
            PropertySourceEnum.ATTRIBUTE: self._get_attribute_value_by_id,
            PropertySourceEnum.CITATION: self._get_citation_value,
            PropertySourceEnum.VALUE: lambda field: field,
        }
        if getter := mapping.get(property_source.source):
            return getter(property_source.field)
        raise ValueError(f"Unsupported property source: {property_source.source}")

    @staticmethod
    def _parse_date_with_formats(value: str, formats: list[str] | None) -> datetime | None:
        if formats:
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    _log.debug("Failed to parse date %r with format %r", value, fmt)
            _log.warning("Failed to parse date %r with any of the formats: %s", value, formats)
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            _log.warning("Failed to parse date %r with ISO format", value)
            return None

    async def updated_at(self, auth_context: AuthContext) -> datetime | None:
        config = getattr(self, "_config", None)
        if not config or not hasattr(config, "updated_at"):
            return None
        for property_source in config.updated_at:
            value = self._get_property_value_by_source(property_source)
            if value and (value := value.strip()):
                if res := self._parse_date_with_formats(value, property_source.formats):
                    return res
        return None

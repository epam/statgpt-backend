import logging
import typing
import uuid
from collections.abc import Iterable

from sdmx.message import StructureMessage
from sdmx.model.common import BaseAnnotation
from sdmx.model.v21 import ContentConstraint
from sdmx.model.v21 import DataflowDefinition as DataFlow

from statgpt.common.data.base import DataSetAvailabilityQuery
from statgpt.common.data.base.property_source import PropertySource, PropertySourceEnum
from statgpt.common.data.base.sdmx_schemas import Sdmx30AnnotationModel
from statgpt.common.data.base.updated_at_mixin import UpdatedAtMixin
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig
from statgpt.common.data.sdmx import Sdmx21DataSet
from statgpt.common.data.sdmx.common import SdmxDimension
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute

if typing.TYPE_CHECKING:
    from statgpt.common.data.quanthub.v21.datasource import QuanthubSdmx21DataSourceHandler


_log = logging.getLogger(__name__)


class QuanthubSdmx21DataSet(UpdatedAtMixin, Sdmx21DataSet):
    def __init__(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: QuanthubDataSetConfig,
        handler: 'QuanthubSdmx21DataSourceHandler',
        dataflow: DataFlow,
        locale: str,
        dimensions: Iterable[SdmxDimension],
        attributes: Iterable[Sdmx21Attribute],
        attribute_values: dict[str, str | None],
        annotations: Iterable[Sdmx30AnnotationModel],
    ):
        super().__init__(
            entity_id=entity_id,
            title=title,
            config=config,
            handler=handler,
            dataflow=dataflow,
            locale=locale,
            dimensions=dimensions,
            attributes=attributes,
        )
        self._config: QuanthubDataSetConfig = config
        self._attribute_values = attribute_values
        self._annotations = list(annotations)

    @property
    def dataset_url(self) -> str | None:
        if self._datasource.config.use_data_explorer_for_dataset_url:  # type: ignore
            if data_explorer_url := self._resolved_data_explorer_base_url():
                return f"{data_explorer_url}?urn={self._short_urn}"
            else:
                _log.warning(
                    "Data explorer URL is not configured on the dataset or data source: %s",
                    self._datasource.source_id,  # type: ignore
                )
        return super().dataset_url

    def _get_annotation_by_id(self, annotation_id: str) -> Sdmx30AnnotationModel | None:
        return next((a for a in self._annotations if a.id == annotation_id), None)

    def _get_annotation_value_by_id(self, annotation_id: str) -> str | None:
        if annotation := self._get_annotation_by_id(annotation_id):
            return annotation.value
        return None

    def _get_attribute_value_by_id(self, attribute_id: str) -> str | None:
        return self._attribute_values.get(attribute_id)

    def _get_citation_value(self, field: str) -> str | None:
        if self._config.citation:
            return getattr(self._config.citation, field, None)
        return None

    def _get_property_value_by_source(self, property_source: PropertySource) -> str | None:
        """Get the property value using the specified property source"""
        mapping = {
            PropertySourceEnum.ANNOTATION: self._get_annotation_value_by_id,
            PropertySourceEnum.ATTRIBUTE: self._get_attribute_value_by_id,
            PropertySourceEnum.CITATION: self._get_citation_value,
            PropertySourceEnum.VALUE: lambda field: field,
        }
        if getter := mapping.get(property_source.source):
            return getter(property_source.field)
        raise ValueError(f"Unsupported property source: {property_source.source}")

    def _availability_result_to_query(
        self, availability_result: StructureMessage
    ) -> DataSetAvailabilityQuery:
        result = super()._availability_result_to_query(availability_result)

        constraints_iterator = iter(availability_result.constraint.values())
        constraint: ContentConstraint | None = next(constraints_iterator, None)

        if constraint is not None and "TIME_PERIOD" not in result:
            start, end = self._parse_time_period_from(constraint.annotations)
            result.time_period_start, result.time_period_end = start, end

        return result

    @staticmethod
    def _parse_time_period_from(annotations: list[BaseAnnotation]) -> tuple[str | None, str | None]:
        start, end = None, None

        for annotation in annotations:
            if annotation.id == "time_period_start":
                start = annotation.value or annotation.title
            elif annotation.id == "time_period_end":
                end = annotation.value or annotation.title

        return start, end

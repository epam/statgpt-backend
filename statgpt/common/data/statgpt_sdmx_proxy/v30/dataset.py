import typing
import uuid
from collections.abc import Iterable

from sdmx.model.v21 import DataflowDefinition as DataFlow

from statgpt.common.data.base.sdmx_schemas import Sdmx30AnnotationModel
from statgpt.common.data.base.updated_at_mixin import UpdatedAtMixin
from statgpt.common.data.sdmx.common import SdmxDataSetQuery, SdmxDimension
from statgpt.common.data.sdmx.common.config import SdmxDataSetConfig
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataSet

if typing.TYPE_CHECKING:
    from .datasource import StatGptSdmxProxyDataSourceHandler as DataSourceHandler


class StatGptSdmxProxyDataSet(UpdatedAtMixin, Sdmx21DataSet):
    """Dataset for StatGPT SDMX proxy (SDMX 3.0 API, parsed as SDMX 2.1 models)."""

    def __init__(
        self,
        entity_id: uuid.UUID,
        title: str,
        config: SdmxDataSetConfig,
        handler: 'DataSourceHandler',
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
        self._attribute_values = attribute_values
        self._annotations = list(annotations)

    def _get_query_params(self, sdmx_query: SdmxDataSetQuery) -> dict:
        return sdmx_query.get_params_v30()

    def _get_annotation_by_id(self, annotation_id: str) -> Sdmx30AnnotationModel | None:
        return next((a for a in self._annotations if a.id == annotation_id), None)

    def _get_annotation_value_by_id(self, annotation_id: str) -> str | None:
        if annotation := self._get_annotation_by_id(annotation_id):
            return annotation.value
        return None

    def _get_attribute_value_by_id(self, attribute_id: str) -> str | None:
        return self._attribute_values.get(attribute_id)

    def _get_citation_value(self, field: str) -> str | None:
        if self.config.citation:
            return getattr(self.config.citation, field, None)
        return None

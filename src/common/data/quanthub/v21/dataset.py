import typing
from collections.abc import Iterable
from datetime import datetime

from sdmx.model.common import DataAttribute
from sdmx.model.v21 import DataflowDefinition as DataFlow

from common.auth.auth_context import AuthContext
from common.data.quanthub.config import QuanthabDataSetConfig
from common.data.quanthub.v21.qh_sdmx_30_schemas import QhAnnotation
from common.data.sdmx import Sdmx21DataSet
from common.data.sdmx.common import SdmxDimension

if typing.TYPE_CHECKING:
    from common.data.quanthub.v21.datasource import QuanthubSdmx21DataSourceHandler


class QuanthubSdmx21DataSet(Sdmx21DataSet):
    def __init__(
        self,
        entity_id: str,
        title: str,
        config: QuanthabDataSetConfig,
        handler: 'QuanthubSdmx21DataSourceHandler',
        dataflow: DataFlow,
        locale: str,
        dimensions: Iterable[SdmxDimension],
        attributes: Iterable[DataAttribute],
        annotations: Iterable[QhAnnotation],
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
        self._config: QuanthabDataSetConfig = config
        self._handler: 'QuanthubSdmx21DataSourceHandler' = handler
        self._annotations = list(annotations)

    def _get_annotation_by_id(self, annotation_id: str) -> QhAnnotation | None:
        return next((a for a in self._annotations if a.id == annotation_id), None)

    async def updated_at(self, auth_context: AuthContext) -> datetime | None:
        annotation = self._get_annotation_by_id(self._config.updated_at_annotation)
        if annotation and annotation.value:
            return datetime.fromisoformat(annotation.value)
        return await super().updated_at(auth_context)

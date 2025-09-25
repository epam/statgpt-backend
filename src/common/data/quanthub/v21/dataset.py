import typing
from collections.abc import Iterable
from datetime import datetime

from sdmx.model.v21 import DataflowDefinition as DataFlow
from sdmx.model.v21 import DataStructureDefinition

from common.auth.auth_context import AuthContext
from common.config.logging import multiline_logger as logger
from common.data.base import DataSetQuery
from common.data.quanthub.config import QuanthubDataSetConfig
from common.data.quanthub.v21.qh_sdmx_30_schemas import QhAnnotation
from common.data.sdmx import Sdmx21DataSet
from common.data.sdmx.common import SdmxDimension
from common.data.sdmx.v21.attribute import Sdmx21Attribute
from common.data.sdmx.v21.dataset import Sdmx21DataResponse
from common.data.sdmx.v21.query import SdmxDataSetQuery
from common.data.sdmx.v21.utils import convert_keys_to_str

if typing.TYPE_CHECKING:
    from common.data.quanthub.v21.datasource import QuanthubSdmx21DataSourceHandler


class QuanthubSdmx21DataSet(Sdmx21DataSet):
    def __init__(
        self,
        entity_id: str,
        title: str,
        config: QuanthubDataSetConfig,
        handler: 'QuanthubSdmx21DataSourceHandler',
        dataflow: DataFlow,
        locale: str,
        dimensions: Iterable[SdmxDimension],
        attributes: Iterable[Sdmx21Attribute],
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
        self._config: QuanthubDataSetConfig = config
        self._annotations = list(annotations)

    def _get_annotation_by_id(self, annotation_id: str) -> QhAnnotation | None:
        return next((a for a in self._annotations if a.id == annotation_id), None)

    async def updated_at(self, auth_context: AuthContext) -> datetime | None:
        annotation = self._get_annotation_by_id(self._config.updated_at_annotation)
        if annotation and annotation.value:
            return datetime.fromisoformat(annotation.value)
        return await super().updated_at(auth_context)

    def _get_data_explorer_url(
        self,
        base_url: str,
        dsd: DataStructureDefinition,
        key_dict: dict[str, list[str]],
        sdmx_query: SdmxDataSetQuery,
    ) -> str:
        params = {
            'urn': self._short_urn,
            'filter': convert_keys_to_str(dsd, key_dict),
        }

        try:
            if td_query := sdmx_query.time_dimension_query:
                start_period = td_query.start_period
                end_period = td_query.end_period
            else:
                start_period = None
                end_period = None

            if start_period:
                params['startPeriod'] = start_period if '-' in start_period else f"{start_period}-A"
            if end_period:
                params['endPeriod'] = end_period if '-' in end_period else f"{end_period}-A"
        except Exception as e:
            logger.exception(e)

        params_str = '&'.join([f'{k}={v}' for k, v in params.items()])
        return f"{base_url}?{params_str}"

    async def query(
        self, query: DataSetQuery, auth_context: AuthContext
    ) -> Sdmx21DataResponse | None:
        sdmx_query, data_msg = await self._query_data(query, auth_context)

        if data_explorer_url := self._datasource.config.get_data_explorer_url():  # type: ignore
            url = self._get_data_explorer_url(
                base_url=data_explorer_url,
                dsd=self._artefact.structure,
                key_dict=sdmx_query.get_key(),
                sdmx_query=sdmx_query,
            )
        else:
            url = self._get_query_url(data_msg.response)  # type: ignore

        sdmx_pandas = self._data_msg_to_dataframe(data_msg)
        if sdmx_pandas.empty:
            req_url = data_msg.response.url if data_msg.response else None
            logger.warning(f"Empty response in dataset(id={self.entity_id}), url={req_url!r}")
            return None

        sdmx_pandas = self._include_attributes(sdmx_pandas)

        return Sdmx21DataResponse(
            dataset=self,
            df=sdmx_pandas,
            sdmx_query=sdmx_query,
            url=url,
        )

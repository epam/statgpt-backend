import logging
import typing
import uuid
from collections.abc import Iterable

import pandas as pd
from sdmx.message import StructureMessage
from sdmx.model.common import BaseAnnotation
from sdmx.model.v21 import ContentConstraint
from sdmx.model.v21 import DataflowDefinition as DataFlow
from sdmx.model.v21 import DataStructureDefinition

from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.data.base import DataResponseStatus, DataSetAvailabilityQuery, DataSetQuery
from statgpt.common.data.base.updated_at_mixin import UpdatedAtMixin
from statgpt.common.data.quanthub.config import QuanthubDataSetConfig
from statgpt.common.data.quanthub.sdmx_schemas.v30 import QhAnnotation
from statgpt.common.data.sdmx import Sdmx21DataSet
from statgpt.common.data.sdmx.common import SdmxDimension
from statgpt.common.data.sdmx.v21.attribute import Sdmx21Attribute
from statgpt.common.data.sdmx.v21.dataset import Sdmx21DataResponse
from statgpt.common.data.sdmx.v21.query import SdmxDataSetQuery, SdmxQueryReadinessStatus
from statgpt.common.data.sdmx.v21.utils import convert_keys_to_str
from statgpt.common.schemas.enums import DataParsingStatus, DataRequestStatus

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
        self._attribute_values = attribute_values
        self._annotations = list(annotations)

    @property
    def dataset_url(self) -> str | None:
        if self._datasource.config.use_data_explorer_for_dataset_url:  # type: ignore
            if data_explorer_url := self._datasource.config.get_data_explorer_url():  # type: ignore
                return f"{data_explorer_url}?urn={self._short_urn}"
            else:
                _log.warning("Data explorer URL is not configured for the data source: %s", self._datasource.source_id)  # type: ignore
        return super().dataset_url

    def _get_annotation_by_id(self, annotation_id: str) -> QhAnnotation | None:
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
            _log.exception(e)

        params_str = '&'.join([f'{k}={v}' for k, v in params.items()])
        return f"{base_url}?{params_str}"

    async def query(
        self, query: DataSetQuery, auth_context: AuthContext
    ) -> Sdmx21DataResponse | None:
        status, missing_dimensions = self._evaluate_data_query_status(query)
        if status != SdmxQueryReadinessStatus.READY:
            raise ValueError(
                f"Query is not ready: {query}, missing dimensions: {missing_dimensions}"
            )
        sdmx_query = self._to_sdmx_query(query)

        try:
            data_msg = await self._query_sdmx_data(sdmx_query, auth_context)
        except Exception as e:
            _log.exception(e)
            return Sdmx21DataResponse(
                dataset=self,
                df=pd.DataFrame(),
                sdmx_query=sdmx_query,
                url=None,
                status=DataResponseStatus(
                    request_status=DataRequestStatus.FAILED,
                    parsing_status=DataParsingStatus.NA,
                ),
            )

        if not data_msg:
            return Sdmx21DataResponse(
                dataset=self,
                df=pd.DataFrame(),
                sdmx_query=sdmx_query,
                url=None,
                status=DataResponseStatus(
                    request_status=DataRequestStatus.SUCCESS,
                    parsing_status=DataParsingStatus.FAILED,
                ),
            )

        if data_explorer_url := self._datasource.config.get_data_explorer_url():  # type: ignore
            url = self._get_data_explorer_url(
                base_url=data_explorer_url,
                dsd=self._artefact.structure,
                key_dict=sdmx_query.get_key(),
                sdmx_query=sdmx_query,
            )
        else:
            url = self._get_query_url(data_msg.response)  # type: ignore

        try:
            sdmx_pandas = await self._data_msg_to_dataframe(data_msg)
            sdmx_pandas = await self._include_attributes(sdmx_pandas)
        except Exception as e:
            _log.exception(e)
            return Sdmx21DataResponse(
                dataset=self,
                df=pd.DataFrame(),
                sdmx_query=sdmx_query,
                url=url,
                status=DataResponseStatus(
                    request_status=DataRequestStatus.SUCCESS,
                    parsing_status=DataParsingStatus.FAILED,
                ),
            )

        return Sdmx21DataResponse(
            dataset=self,
            df=sdmx_pandas,
            sdmx_query=sdmx_query,
            url=url,
            status=DataResponseStatus(
                request_status=DataRequestStatus.SUCCESS,
                parsing_status=DataParsingStatus.SUCCESS,
            ),
        )

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

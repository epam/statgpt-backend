import typing as t
from abc import ABC

from httpx import HTTPStatusError
from langchain_core.documents import Document
from sdmx.message import StructureMessage
from sdmx.model.v21 import DataflowDefinition as Dataflow

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.base import (
    DataSetDescriptor,
    DataSourceHandler,
    DataSourceType,
    VirtualDimensionCategory,
)
from common.data.sdmx.common import (
    ComplexIndicator,
    DimensionCodeCategory,
    SdmxConstants,
    SdmxDataSetConfig,
    SdmxDataSourceConfig,
    UrnParser,
)
from common.data.sdmx.v21.dataflow_loader import DataflowLoader
from common.data.sdmx.v21.dataset import Sdmx21DataSet, SdmxOfflineDataSet
from common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from common.data.sdmx.v21.sdmx_client import AsyncSdmxClient

from .schemas import Urn


class Sdmx21DataSourceHandler(
    DataSourceHandler[SdmxDataSourceConfig, Sdmx21DataSet | SdmxOfflineDataSet, SdmxDataSetConfig],
    ABC,
):
    def __init__(self, config: SdmxDataSourceConfig):
        super().__init__(config)
        self._urn_parser = UrnParser.create_default()

    @staticmethod
    def data_source_type() -> DataSourceType:
        return DataSourceType(
            type_id="SDMX21", name="SDMX 2.1 Registry", description="SDMX 2.1 Registry data source"
        )

    @staticmethod
    def parse_config(d: dict) -> SdmxDataSourceConfig:
        return SdmxDataSourceConfig.model_validate(d)

    @staticmethod
    def parse_data_set_config(d: dict) -> SdmxDataSetConfig:
        return SdmxDataSetConfig.model_validate(d)

    @property
    def source_id(self) -> str:
        return self._config.get_id()

    @property
    def name(self) -> str:
        return self._config.get_name()

    @property
    def description(self) -> t.Optional[str]:
        return self._config.description

    def create_sdmx_client(self, auth_context: AuthContext) -> AsyncSdmxClient:
        return AsyncSdmxClient.from_config(self._config, auth_context)

    async def is_dataset_available(self, config: dict, auth_context: AuthContext) -> bool:
        if auth_context.is_system:
            return True
        else:
            try:
                urn = self._urn_parser.parse(config["urn"])
                client = self.create_sdmx_client(auth_context)
                await client.availableconstraint(
                    agency_id=urn.agency_id,
                    resource_id=urn.resource_id,
                    version=urn.version if urn.version else "latest",
                    params={"references": "none"},
                    use_cache=False,
                )
                return True
            except HTTPStatusError as e:
                # availability endpoint returns 400 with NotFound instead of 403
                # treat 400 as Forbidden as well
                if e.response.status_code in [403, 400]:
                    # 403 means user doesn't have access to dataset
                    return False
                else:
                    raise e

    async def list_datasets(self, auth_context: AuthContext) -> list[DataSetDescriptor]:
        client = self.create_sdmx_client(auth_context)

        message: StructureMessage = await client.dataflow(
            agency_id="all", resource_id="all", version="latest", params={}
        )
        dataflows: list[Dataflow] = list(message.dataflow.values())
        return [
            DataSetDescriptor(
                source_id=self._urn_parser.parse(dataflow.urn).get_short_urn(),
                name=dataflow.name[self._config.locale],
                description=dataflow.description.localizations.get(self._config.locale),
                details=SdmxDataSetConfig(
                    urn=self._urn_parser.parse(dataflow.urn).get_short_urn(),
                    indicatorDimensions=["INDICATOR"],  # type: ignore
                ).model_dump(by_alias=True),
            )
            for dataflow in dataflows
        ]

    @property
    def entity_id(self) -> str:
        return self._config.get_id()

    async def get_dataset(
        self,
        entity_id: str,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
    ) -> Sdmx21DataSet | SdmxOfflineDataSet:
        dataset_config = self.parse_data_set_config(config)
        _urn = self._urn_parser.parse(dataset_config.urn)
        urn = Urn(
            agency_id=_urn.agency_id,
            resource_id=_urn.resource_id,
            version=_urn.version if _urn.version else "latest",
        )

        sdmx_client = self.create_sdmx_client(auth_context)

        try:
            dataflow_loader = DataflowLoader(sdmx_client)
            structure_message = await dataflow_loader.load_structure_message(urn)
        except Exception as e:
            if allow_offline:
                msg = "Failed to load the dataflow or its associated structures."
                logger.exception(msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, status_details=msg)
            else:
                raise e

        try:
            dimensions_creator = DimensionsCreator(structure_message, urn, self._config.locale)
            dimensions = await dimensions_creator.create_dimensions()
        except Exception as e:
            if allow_offline:
                msg = "Failed to create dimensions from the loaded structure message."
                logger.exception(msg)
                return SdmxOfflineDataSet(entity_id, title, dataset_config, status_details=msg)
            else:
                raise e

        try:
            dataflow = structure_message.dataflow[urn]
            return Sdmx21DataSet(
                entity_id=entity_id,
                title=title,
                config=dataset_config,
                handler=self,
                dataflow=dataflow,
                locale=self._config.locale,
                dimensions=dimensions,
                attributes=dataflow.structure.attributes,
            )
        except Exception as e:
            if allow_offline:
                msg = "Failed to create dataset class."
                logger.exception(f"{msg}. See exception details below.")
                msg += (
                    " Probably there is a mistake in configuration."
                    " For example, the indicator dimension name is incorrect."
                )
                return SdmxOfflineDataSet(entity_id, title, dataset_config, status_details=msg)
            else:
                raise e

    async def close(self):
        # do nothing
        pass

    async def get_indicator_from_document(self, document: Document) -> ComplexIndicator:
        return ComplexIndicator.from_document(document)

    async def document_to_dimension_category(
        self, document: Document
    ) -> DimensionCodeCategory | VirtualDimensionCategory:
        if SdmxConstants.METADATA_DIMENSION_ID in document.metadata:
            return DimensionCodeCategory.from_document(document)
        else:
            return VirtualDimensionCategory.from_document(document)

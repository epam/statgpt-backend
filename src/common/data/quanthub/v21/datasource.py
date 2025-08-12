import httpx

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.data.base import DataSourceType
from common.data.quanthub.config import QuanthabDataSetConfig, QuanthubSdmxDataSourceConfig
from common.data.quanthub.v21.dataset import QuanthubSdmx21DataSet
from common.data.sdmx import Sdmx21DataSourceHandler
from common.data.sdmx.v21.dataflow_loader import DataflowLoader
from common.data.sdmx.v21.dataset import SdmxOfflineDataSet
from common.data.sdmx.v21.dimensions_creator import DimensionsCreator
from common.data.sdmx.v21.schemas import Urn

from .qh_sdmx_client import AsyncQuanthubClient


# (DataSourceHandler[SdmxDataSourceConfig, InMemorySdmx21DataSet | SdmxOfflineDataSet, QuanthubDataSetConfig],ABC)
# todo: add generic typing with QuanthubInMemorySdmx21DataSet
class QuanthubSdmx21DataSourceHandler(Sdmx21DataSourceHandler):
    def __init__(self, config: QuanthubSdmxDataSourceConfig):
        super().__init__(config)
        self._config: QuanthubSdmxDataSourceConfig = config  # for type hinting

    def create_sdmx_client(self, auth_context: AuthContext) -> AsyncQuanthubClient:
        return AsyncQuanthubClient.from_config(self._config, auth_context)

    async def get_dataset(
        self,
        entity_id: str,
        title: str,
        config: dict,
        auth_context: AuthContext,
        allow_offline: bool = False,
    ) -> QuanthubSdmx21DataSet | SdmxOfflineDataSet:
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
            annotations = await sdmx_client.dynamic_dataflow_annotations(
                agency_id=urn.agency_id, resource_id=urn.resource_id, version=urn.version
            )
        except httpx.RequestError as e:
            logger.exception(
                f"Failed to load annotations for the dataflow({urn})."
                f"\nRequest: {e.request.method} {e.request.url}"
                + (f"\nContent: {e.request.content!r}" if e.request.content else "")
            )
            annotations = []
        except Exception:
            logger.exception(f"Failed to load annotations for the dataflow({urn}).")
            annotations = []

        try:
            dataflow = structure_message.dataflow[urn]
            return QuanthubSdmx21DataSet(
                entity_id=entity_id,
                title=title,
                config=dataset_config,
                handler=self,
                dataflow=dataflow,
                locale=self._config.locale,
                dimensions=dimensions,
                attributes=dataflow.structure.attributes,
                annotations=annotations,
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

    @staticmethod
    def data_source_type() -> DataSourceType:
        return DataSourceType(
            type_id="QH_SDMX21",
            name="Quanthub SDMX 2.1 Registry",
            description="Quanthub SDMX 2.1 Registry data source",
        )

    @staticmethod
    def parse_config(d: dict) -> QuanthubSdmxDataSourceConfig:
        return QuanthubSdmxDataSourceConfig.model_validate(d)

    @staticmethod
    def parse_data_set_config(d: dict) -> QuanthabDataSetConfig:
        return QuanthabDataSetConfig.model_validate(d)

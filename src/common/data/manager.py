from .base.datasource import DataSourceConfig, DataSourceHandler
from .quanthub.config import QuanthubSdmxDataSourceConfig
from .quanthub.v21.datasource import QuanthubSdmx21DataSourceHandler
from .sdmx import Sdmx21DataSourceHandler
from .sdmx.common import SdmxDataSourceConfig


class DataManager:
    _config_mapping: dict[str, type[DataSourceConfig]] = {
        "SDMX21": SdmxDataSourceConfig,
        "QH_SDMX21": QuanthubSdmxDataSourceConfig,
    }

    _handlers_mapping: dict[str, type[DataSourceHandler]] = {
        "SDMX21": Sdmx21DataSourceHandler,
        "QH_SDMX21": QuanthubSdmx21DataSourceHandler,
    }

    @classmethod
    def get_config_class(cls, data_source_name) -> type[DataSourceConfig]:
        if data_source_name not in cls._config_mapping:
            raise ValueError(f"DataSourceConfig not found for '{data_source_name}' data source.")

        return cls._config_mapping[data_source_name]

    @classmethod
    def get_data_source_handler_class(cls, data_source_name) -> type[DataSourceHandler]:
        if data_source_name not in cls._handlers_mapping:
            raise ValueError(f"DataSourceHandler class not found for '{data_source_name}")

        return cls._handlers_mapping[data_source_name]

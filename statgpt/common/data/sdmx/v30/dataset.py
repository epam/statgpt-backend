from statgpt.common.data.quanthub.v21.dataset import QuanthubSdmx21DataSet
from statgpt.common.data.sdmx.v21.query import SdmxDataSetQuery


class Sdmx30ProxyDataSet(QuanthubSdmx21DataSet):
    """Proxy SDMX 3.0 dataset parsed via sdmx1 (SDMX 2.1) models."""

    def _get_query_params(self, sdmx_query: SdmxDataSetQuery) -> dict:
        params = sdmx_query.get_params()
        if "detail" in params:
            params = dict(params)
            params.pop("detail", None)
        return params

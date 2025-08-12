import os
from io import BytesIO
from typing import AnyStr

from sdmx.message import StructureMessage
from sdmx.reader.xml import Reader

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def read_dsd(path: AnyStr) -> StructureMessage:
    with open(path) as f:
        response_content = f.read()

    response_io = BytesIO(response_content.encode("utf-8"))
    return Reader().convert(response_io)


class AsyncSdmxClientMock:

    async def dataflow(
        self, *, agency_id: str, resource_id: str, version: str, **kwargs
    ) -> StructureMessage:
        mock_dataflows = {
            '3.0.1': "dataflow_IMF.STA-CPI(3.0.1).xml",
            '4.0.0': "dataflow_IMF.STA-CPI(4.0.0).xml",
        }

        short_urn = f"{agency_id}:{resource_id}({version})"
        if mocked_file := mock_dataflows.get(version):
            return read_dsd(os.path.join(CURRENT_DIR, "data", mocked_file))
        raise ValueError(f"Unknown dataflow: {short_urn!r} {kwargs=}")

    async def conceptscheme(
        self, *, agency_id: str, resource_id: str, version: str, **kwargs
    ) -> StructureMessage:
        mocked_conceptschemes = {
            "IMF:CS_MASTER_SYSTEM(1.0.2)": "conceptscheme_IMF-CS_MASTER_SYSTEM(1.0.2).xml",
            "IMF:CS_MASTER_DATA(1.0.0)": "conceptscheme_IMF-CS_MASTER_DATA(1.0.0).xml",
            "IMF:CS_MASTER_DOMAIN(1.0.0)": "conceptscheme_IMF-CS_MASTER_DOMAIN(1.0.0).xml",
            "IMF:CS_MASTER(2.1.0)": "conceptscheme_IMF-CS_MASTER(2.1.0).xml",
            "IMF.STA:CS_CPI(4.0.0)": "conceptscheme_IMF.STA-CS_CPI(4.0.0).xml",
            "IMF.STA:CS_CPI(3.0.0)": "conceptscheme_IMF.STA-CS_CPI(3.0.0).xml",
            "IMF.STA:CS_MASTER(1.0.0)": "conceptscheme_IMF.STA-CS_MASTER(1.0.0).xml",
            "SDMX:SDMX_CONCEPT_ROLES(1.0)": "conceptscheme_SDMX-SDMX_CONCEPT_ROLES(1.0).xml",
        }

        short_urn = f"{agency_id}:{resource_id}({version})"
        if mocked_file := mocked_conceptschemes.get(short_urn):
            return read_dsd(os.path.join(CURRENT_DIR, "data", mocked_file))
        raise ValueError(f"Unknown concept scheme: {short_urn!r} {kwargs=}")

    async def codelist(
        self, *, agency_id: str, resource_id: str, version: str, **kwargs
    ) -> StructureMessage:
        mocked_codelists = {
            "IMF:CL_TOPIC(2.2.0)": "codelist_IMF-CL_TOPIC(2.2.0).xml",
            "IMF:CL_METHODOLOGY(2.5.0)": "codelist_IMF-CL_METHODOLOGY(2.5.0).xml",
            "IMF:CL_INDEX_TYPE(2.6.0)": "codelist_IMF-CL_INDEX_TYPE(2.6.0).xml",
            "IMF:CL_DECIMALS(1.0.2)": "codelist_IMF-CL_DECIMALS(1.0.2).xml",
            "IMF:CL_TRANSFORMATION(2.19.0)": "codelist_IMF-CL_TRANSFORMATION(2.19.0).xml",
            "IMF:CL_UNIT(2.8.0)": "codelist_IMF-CL_UNIT(2.8.0).xml",
            "IMF:CL_UNIT_MULT(1.0.2)": "codelist_IMF-CL_UNIT_MULT(1.0.2).xml",
            "IMF:CL_COUNTRY(1.5.1)": "codelist_IMF-CL_COUNTRY(1.5.1).xml",
            "IMF:CL_COICOP_1999(1.0.1)": "codelist_IMF-CL_COICOP_1999(1.0.1).xml",
            "IMF:CL_LANGUAGE(1.1.0)": "codelist_IMF-CL_LANGUAGE(1.1.0).xml",
            "IMF:CL_REPORTING_PERIOD_TYPE(1.3.0)": "codelist_IMF-CL_REPORTING_PERIOD_TYPE(1.3.0).xml",
            "IMF.STA:CL_CPI_TYPE_OF_TRANSFORMATION(4.0.0)": "codelist_IMF.STA-CL_CPI_TYPE_OF_TRANSFORMATION(4.0.0).xml",
            "IMF.STA:CL_CPI_TYPE_OF_TRANSFORMATION(3.1.0)": "codelist_IMF.STA-CL_CPI_TYPE_OF_TRANSFORMATION(3.1.0).xml",
            "IMF:CL_ACCESS_SHARING_LEVEL(1.0.2)": "codelist_IMF-CL_ACCESS_SHARING_LEVEL(1.0.2).xml",
            "IMF:CL_DERIVATION_TYPE(1.2.1)": "codelist_IMF-CL_DERIVATION_TYPE(1.2.1).xml",
            "IMF:CL_SEC_CLASSIFICATION(1.0.1)": "codelist_IMF-CL_SEC_CLASSIFICATION(1.0.1).xml",
            "IMF:CL_OVERLAP(1.0.0)": "codelist_IMF-CL_OVERLAP(1.0.0).xml",
            "IMF:CL_ORGANIZATION(2.1.0)": "codelist_IMF-CL_ORGANIZATION(2.1.0).xml",
            "IMF:CL_DEPARTMENT(1.0.2)": "codelist_IMF-CL_DEPARTMENT(1.0.2).xml",
            "IMF:CL_FREQ(1.0.3)": "codelist_IMF-CL_FREQ(1.0.3).xml",
        }

        short_urn = f"{agency_id}:{resource_id}({version})"
        if mocked_file := mocked_codelists.get(short_urn):
            return read_dsd(os.path.join(CURRENT_DIR, "data", mocked_file))
        raise ValueError(f"Unknown codelist: {short_urn!r} {kwargs=}")

    async def availableconstraint(
        self, *, agency_id: str, resource_id: str, version: str, **kwargs
    ) -> StructureMessage:
        mocked_constraints = {
            "IMF.STA:CPI(4.0.0)": "availableconstraint_IMF.STA,CPI,4.0.0.xml",
            "IMF.STA:CPI(3.0.1)": "availableconstraint_IMF.STA,CPI,3.0.1.xml",
        }

        short_urn = f"{agency_id}:{resource_id}({version})"
        if mocked_file := mocked_constraints.get(short_urn):
            return read_dsd(os.path.join(CURRENT_DIR, "data", mocked_file))
        raise ValueError(f"Unknown availableconstraint: {short_urn!r} {kwargs=}")


class BackgroundTasksMock:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))

"""Migrate collections

Revision ID: 9c86e3f19f2c
Revises: 78b05c24c076
Create Date: 2024-05-27 18:05:18.012772

"""

import dataclasses
import re
import uuid
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9c86e3f19f2c'
down_revision: Union[str, None] = '78b05c24c076'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


@dataclasses.dataclass
class CodeListMapping:
    old_name: str
    collection_name: str
    source: str


CODE_LISTS: list[CodeListMapping] = [
    CodeListMapping(
        old_name="SDMX_CodeList_BIS_CL_AREA_1_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=BIS:CL_AREA(1.1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_BIS_CL_BIS_UNIT_1_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=BIS:CL_BIS_UNIT(1.1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_BIS_CL_FREQ_1_0_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=BIS:CL_FREQ(1.0.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_ECB_CL_CURRENCY_1_0_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=ECB:CL_CURRENCY(1.0.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_ECB_CL_EXR_SUFFIX_1_0_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=ECB:CL_EXR_SUFFIX(1.0.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_ECB_CL_EXR_TYPE_1_0_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=ECB:CL_EXR_TYPE(1.0.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_ECB_CL_FREQ_1_0_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=ECB:CL_FREQ(1.0.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_ESTAT_FREQ_1_0_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=ESTAT:FREQ(1.0.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_ESTAT_GEO_1_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=ESTAT:GEO(1.1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_ESTAT_NA_ITEM_1_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=ESTAT:NA_ITEM(1.1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_ESTAT_UNIT_1_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=ESTAT:UNIT(1.1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_FRB_CL_BG_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=FRB:CL_BG(1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_FRB_CL_CATEGORY_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=FRB:CL_CATEGORY(1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_FRB_CL_H8_UNITS_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=FRB:CL_H8_UNITS(1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_FRB_CL_ITEM_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=FRB:CL_ITEM(1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_FRB_CL_SA_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=FRB:CL_SA(1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_CL_COUNTRIES_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=IMF:CL_COUNTRIES(1.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_CL_COUNTRY_1_9_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=IMF:CL_COUNTRY(1.9.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_CL_FREQ_1_0_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=IMF:CL_FREQ(1.0.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_RES_CL_WEO_INDICATOR_1_0_4_BAAI_bge_base_en_v",
        collection_name="CodeList=IMF_RES:CL_WEO_INDICATOR(1.0.4)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_STA_CL_BOP_BP6_INDICATOR_1_0_1_BAAI_bge_base_",
        collection_name="CodeList=IMF_STA:CL_BOP_BP6_INDICATOR(1.0.1)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_STA_CL_CPI_INDICATOR_1_0_1_BAAI_bge_base_en_v",
        collection_name="CodeList=IMF_STA:CL_CPI_INDICATOR(1.0.1)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_STA_CL_GFS_INDICATOR_1_1_1_BAAI_bge_base_en_v",
        collection_name="CodeList=IMF_STA:CL_GFS_INDICATOR(1.1.1)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_STA_CL_GFS_SECTOR_1_0_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=IMF_STA:CL_GFS_SECTOR(1.0.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_STA_CL_IFS_INDICATOR_1_0_1_BAAI_bge_base_en_v",
        collection_name="CodeList=IMF_STA:CL_IFS_INDICATOR(1.0.1)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_STA_CL_MFS_INDICATOR_1_1_0_BAAI_bge_base_en_v",
        collection_name="CodeList=IMF_STA:CL_MFS_INDICATOR(1.1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_IMF_STA_CL_SNA_INDICATOR_1_1_0_BAAI_bge_base_en_v",
        collection_name="CodeList=IMF_STA:CL_SNA_INDICATOR(1.1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_CL_ACTIVITY_ISIC4_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=OECD:CL_ACTIVITY_ISIC4(1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_CL_ADJUSTMENT_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=OECD:CL_ADJUSTMENT(1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_CL_AREA_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=OECD:CL_AREA(1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_CL_POLLUTANT_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=OECD:CL_POLLUTANT(1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_CL_UNIT_MEASURE_1_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=OECD:CL_UNIT_MEASURE(1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_SDD_NAD_SEEA_CL_BRIDGE_ITEM_1_2_BAAI_bge_bas",
        collection_name="CodeList=OECD.SDD.NAD.SEEA:CL_BRIDGE_ITEM(1.2)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_SDD_NAD_SEEA_CL_MEASURE_1_0_BAAI_bge_base_en",
        collection_name="CodeList=OECD.SDD.NAD.SEEA:CL_MEASURE(1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_SDD_NAD_SEEA_CL_METHODOLOGY_1_0_BAAI_bge_bas",
        collection_name="CodeList=OECD.SDD.NAD.SEEA:CL_METHODOLOGY(1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_OECD_SDD_NAD_SEEA_CL_SOURCE_1_0_BAAI_bge_base_en_",
        collection_name="CodeList=OECD.SDD.NAD.SEEA:CL_SOURCE(1.0)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_SDMX_CL_FREQ_2_0_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=SDMX:CL_FREQ(2.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_SDMX_CL_FREQ_2_1_BAAI_bge_base_en_v1_5",
        collection_name="CodeList=SDMX:CL_FREQ(2.1)",
        source="SDMX21_QH_UAT",
    ),
    CodeListMapping(
        old_name="SDMX_CodeList_WB_CL_WORLD_BANK_WORLD_DEVELOPMENT_INDICATORS_1_0",
        collection_name="CodeList=WB:CL_WORLD_BANK_WORLD_DEVELOPMENT_INDICATORS(1.0.0)",
        source="SDMX21_QH_UAT_GLOBAL_DATA",
    ),
]


def get_all_tables(conn: sa.Connection, table_schema: str) -> list[str]:
    res = conn.execute(
        sa.text(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema='{table_schema}'"
        )
    )
    return res.scalars().all()  # type: ignore


def add_collection(conn: sa.Connection, collection_name: str, source: str | None = None) -> str:
    res = conn.execute(
        sa.text(
            "INSERT INTO collections._names (uuid, collection_name, embedding_model_name, datasource)"
            " VALUES (:uuid, :collection_name, :embedding_model_name, :datasource) RETURNING uuid;"
        ),
        {
            'uuid': uuid.uuid4(),
            'collection_name': collection_name,
            'embedding_model_name': 'text-embedding-3-large',
            'datasource': source,
        },
    )
    return str(res.scalar_one())


def rename_table(old_schema: str, old_name: str, new_schema: str, new_name: str) -> None:
    print(f"Rename table '{old_schema}.{old_name}' to '{new_schema}.{new_name}'")

    op.execute(f'ALTER TABLE {old_schema}."{old_name}" SET SCHEMA {new_schema}')
    op.execute(f'ALTER TABLE {new_schema}."{old_name}" RENAME TO "{new_name}"')


def upgrade() -> None:
    conn = op.get_bind()

    tables = get_all_tables(conn, table_schema='public')
    print(f"{tables=}")

    indicators_regex = re.compile(r'(Indicators_\d+)_BAAI_bge_base_en_v1_5')

    for table in tables:
        if match := indicators_regex.fullmatch(table):
            collection_name = match.group(1)
            collection_uuid = add_collection(conn, collection_name)

            rename_table(
                old_schema='public',
                old_name=table,
                new_schema='collections',
                new_name=f"c_{collection_uuid}",
            )
            rename_table(
                old_schema='public',
                old_name=f'Mapping_{table}',
                new_schema='collections',
                new_name=f"c_{collection_uuid}_mapping",
            )

    for code_list in CODE_LISTS:
        if code_list.old_name in tables:
            collection_uuid = add_collection(conn, code_list.collection_name, code_list.source)

            rename_table(
                old_schema='public',
                old_name=code_list.old_name,
                new_schema='collections',
                new_name=f"c_{collection_uuid}",
            )


def downgrade() -> None:
    conn = op.get_bind()

    res = conn.execute(sa.text("SELECT uuid, collection_name FROM collections._names"))

    for collection_uuid, collection_name in res.all():
        if collection_name.startswith('Indicators_'):
            rename_table(
                old_schema='collections',
                old_name=f"c_{collection_uuid}",
                new_schema='public',
                new_name=f'{collection_name}_BAAI_bge_base_en_v1_5'[:63],
            )
            rename_table(
                old_schema='collections',
                old_name=f"c_{collection_uuid}_mapping",
                new_schema='public',
                new_name=f'Mapping_{collection_name}_BAAI_bge_base_en_v1_5'[:63],
            )
        elif collection_name.startswith('CodeList='):
            code_list = next(m for m in CODE_LISTS if m.collection_name == collection_name)
            rename_table(
                old_schema='collections',
                old_name=f"c_{collection_uuid}",
                new_schema='public',
                new_name=code_list.old_name,
            )
        else:
            print(f"[ERROR] Unknown collection: {collection_name} (uuid={collection_uuid})")

    op.execute("TRUNCATE collections._names")

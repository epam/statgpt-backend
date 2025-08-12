"""add qh datasource type

Revision ID: f6f95fda8420
Revises: 318707f46c84
Create Date: 2025-06-23 03:28:54.735990

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f6f95fda8420'
down_revision: Union[str, None] = '318707f46c84'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("INSERT INTO data_source_types(name, description) VALUES ('QH_SDMX21', '');")
    op.execute(
        "UPDATE data_sources SET type_id = (SELECT id FROM data_source_types WHERE name = 'QH_SDMX21')"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE data_sources SET type_id = (SELECT id FROM data_source_types WHERE name = 'SDMX21')"
    )
    op.execute("DELETE FROM data_source_types WHERE name = 'QH_SDMX21';")

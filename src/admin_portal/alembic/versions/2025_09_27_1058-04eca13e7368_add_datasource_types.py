"""Add datasource types

Revision ID: 04eca13e7368
Revises: fa6b0a97e522
Create Date: 2025-09-27 10:58:36.804657

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '04eca13e7368'
down_revision: Union[str, None] = 'fa6b0a97e522'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("INSERT INTO data_source_types(name, description) VALUES ('SDMX21', '');")
    op.execute("INSERT INTO data_source_types(name, description) VALUES ('QH_SDMX21', '');")


def downgrade() -> None:
    op.execute("DELETE FROM data_source_types WHERE name = 'SDMX21';")
    op.execute("DELETE FROM data_source_types WHERE name = 'QH_SDMX21';")

"""Add Proxy SDMX 3.0 datasource type

Revision ID: 2f0f6f2f0d7b
Revises: 109220bfa072
Create Date: 2026-01-21 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '2f0f6f2f0d7b'
down_revision: Union[str, None] = 'c7f068b2d47d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("INSERT INTO data_source_types(name, description) VALUES ('PROXY_SDMX30', '');")


def downgrade() -> None:
    op.execute("DELETE FROM data_source_types WHERE name = 'PROXY_SDMX30';")

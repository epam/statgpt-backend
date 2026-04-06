"""Add Proxy SDMX 3.0 datasource type

Revision ID: 2f0f6f2f0d7b
Revises: 2263b2983f56
Create Date: 2026-04-02 12:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '2f0f6f2f0d7b'
down_revision: str | None = '2263b2983f56'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("INSERT INTO data_source_types(name, description) VALUES ('PROXY_SDMX30', '');")


def downgrade() -> None:
    op.execute("DELETE FROM data_source_types WHERE name = 'PROXY_SDMX30';")

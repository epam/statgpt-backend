"""add indexing_stats to channel_dataset_versions

Revision ID: 331d3d5b3b30
Revises: 3b8a6a40f1cd
Create Date: 2026-02-19 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '331d3d5b3b30'
down_revision: str | None = '3b8a6a40f1cd'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'channel_dataset_versions',
        sa.Column('indexing_stats', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('channel_dataset_versions', 'indexing_stats')

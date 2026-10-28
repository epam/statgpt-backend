"""summarize discovery indicator coverage

Revision ID: 492ab5673baa
Revises: d7339387d7fc
Create Date: 2026-10-28 12:27:47.117686

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '492ab5673baa'
down_revision: str | None = 'd7339387d7fc'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # No backfill: a NULL source hash means "derive on the next run", which is what every
    # existing record needs.
    op.add_column(
        'discovery_datasets',
        sa.Column('indicators_summary_source_hash', sa.String(), nullable=True),
    )
    op.add_column(
        'discovery_datasets',
        sa.Column('indicators_summarized_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'discovery_indexing_jobs', sa.Column('records_summarized', sa.Integer(), nullable=True)
    )
    op.add_column(
        'discovery_indexing_jobs', sa.Column('records_failed', sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('discovery_indexing_jobs', 'records_failed')
    op.drop_column('discovery_indexing_jobs', 'records_summarized')
    op.drop_column('discovery_datasets', 'indicators_summarized_at')
    op.drop_column('discovery_datasets', 'indicators_summary_source_hash')

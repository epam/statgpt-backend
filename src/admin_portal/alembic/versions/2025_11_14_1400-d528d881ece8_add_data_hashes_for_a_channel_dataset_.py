"""Add data hashes for a channel dataset version

Revision ID: d528d881ece8
Revises: 10fe795dc09d
Create Date: 2025-11-14 14:00:16.313951

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd528d881ece8'
down_revision: str | None = '10fe795dc09d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'channel_dataset_versions',
        sa.Column(
            'structure_metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default=None,
        ),
    )
    op.add_column(
        'channel_dataset_versions',
        sa.Column('structure_hash', sa.String(length=10), nullable=True, server_default=None),
    )
    op.add_column(
        'channel_dataset_versions',
        sa.Column(
            'indicator_dimensions_hash', sa.String(length=10), nullable=True, server_default=None
        ),
    )
    op.add_column(
        'channel_dataset_versions',
        sa.Column(
            'non_indicator_dimensions_hash',
            sa.String(length=10),
            nullable=True,
            server_default=None,
        ),
    )
    op.add_column(
        'channel_dataset_versions',
        sa.Column(
            'special_dimensions_hash', sa.String(length=10), nullable=True, server_default=None
        ),
    )


def downgrade() -> None:
    op.drop_column('channel_dataset_versions', 'special_dimensions_hash')
    op.drop_column('channel_dataset_versions', 'non_indicator_dimensions_hash')
    op.drop_column('channel_dataset_versions', 'indicator_dimensions_hash')
    op.drop_column('channel_dataset_versions', 'structure_hash')
    op.drop_column('channel_dataset_versions', 'structure_metadata')

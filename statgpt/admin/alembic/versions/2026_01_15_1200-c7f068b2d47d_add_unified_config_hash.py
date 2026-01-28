"""Replace legacy config hashes with unified indexing_config_hash

Revision ID: c7f068b2d47d
Revises: 77c67d91641c
Create Date: 2026-01-15 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c7f068b2d47d'
down_revision: str | None = '77c67d91641c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add new unified config hash
    op.add_column(
        'channel_dataset_versions',
        sa.Column('indexing_config_hash', sa.String(length=10), nullable=True),
    )
    # Drop legacy separate config hashes
    op.drop_column('channel_dataset_versions', 'indicators_config_hash')
    op.drop_column('channel_dataset_versions', 'non_indicators_config_hash')
    op.drop_column('channel_dataset_versions', 'special_dimensions_config_hash')


def downgrade() -> None:
    # Re-add legacy config hashes
    op.add_column(
        'channel_dataset_versions',
        sa.Column('indicators_config_hash', sa.String(length=10), nullable=True),
    )
    op.add_column(
        'channel_dataset_versions',
        sa.Column('non_indicators_config_hash', sa.String(length=10), nullable=True),
    )
    op.add_column(
        'channel_dataset_versions',
        sa.Column('special_dimensions_config_hash', sa.String(length=10), nullable=True),
    )
    # Drop unified config hash
    op.drop_column('channel_dataset_versions', 'indexing_config_hash')

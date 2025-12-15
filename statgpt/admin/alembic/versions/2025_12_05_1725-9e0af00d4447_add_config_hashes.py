"""Add config hashes to the `channel_dataset_versions` table

Revision ID: 9e0af00d4447
Revises: 65c149c7db9e
Create Date: 2025-11-24 16:27:10.523922

Manually updated (down_revision): 2025-12-05 17:25:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9e0af00d4447'
down_revision: str | None = '65142518fe0c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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


def downgrade() -> None:
    op.drop_column('channel_dataset_versions', 'special_dimensions_config_hash')
    op.drop_column('channel_dataset_versions', 'non_indicators_config_hash')
    op.drop_column('channel_dataset_versions', 'indicators_config_hash')

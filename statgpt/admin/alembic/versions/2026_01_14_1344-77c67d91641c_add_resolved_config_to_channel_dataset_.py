"""add resolved_config to channel_dataset_versions

Revision ID: 77c67d91641c
Revises: 109220bfa072
Create Date: 2026-01-14 13:44:34.851958

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '77c67d91641c'
down_revision: str | None = '109220bfa072'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'channel_dataset_versions',
        sa.Column('resolved_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('channel_dataset_versions', 'resolved_config')

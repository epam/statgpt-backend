"""Add cleanup status to channel dataset table

Revision ID: 65c149c7db9e
Revises: d528d881ece8
Create Date: 2025-11-14 16:33:52.089023

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '65c149c7db9e'
down_revision: str | None = 'c64458e37902'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'channel_datasets',
        sa.Column(
            'clearing_status',
            postgresql.ENUM(name='preprocessingstatusenum', create_type=False),
            nullable=False,
            server_default='NOT_STARTED',
        ),
    )


def downgrade() -> None:
    op.drop_column('channel_datasets', 'clearing_status')

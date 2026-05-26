"""add deduplication_jobs table

Revision ID: 85c92803e84c
Revises: e7c26f313212
Create Date: 2026-05-26 13:26:23.809373

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '85c92803e84c'
down_revision: str | None = 'e7c26f313212'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'deduplication_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column(
            'status',
            postgresql.ENUM(name='preprocessingstatusenum', create_type=False),
            nullable=False,
        ),
        sa.Column('reason_for_failure', sa.String(), nullable=True),
        sa.Column('non_indicator_remapped', sa.Integer(), nullable=True),
        sa.Column('non_indicator_deleted', sa.Integer(), nullable=True),
        sa.Column('special_remapped', sa.Integer(), nullable=True),
        sa.Column('special_deleted', sa.Integer(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_index(
        'ix_deduplication_jobs_channel_id',
        'deduplication_jobs',
        ['channel_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_deduplication_jobs_channel_id', table_name='deduplication_jobs')
    op.drop_table('deduplication_jobs')

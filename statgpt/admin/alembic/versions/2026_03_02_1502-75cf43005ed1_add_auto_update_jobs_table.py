"""add_auto_update_jobs_table

Revision ID: 75cf43005ed1
Revises: 331d3d5b3b30
Create Date: 2026-02-03 16:08:14.339217
Updated Date: 2026-03-02 15:02:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '75cf43005ed1'
down_revision: str | None = '331d3d5b3b30'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    autoupdateresult_enum = postgresql.ENUM(
        'NO_COMPLETED_VERSION',
        'NO_CHANGES',
        'CONFIG_INCOMPATIBLE',
        'CONFIG_UPDATED',
        'REINDEX_TRIGGERED',
        name='autoupdateresult',
        create_type=False,
    )
    autoupdateresult_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        'auto_update_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('channel_dataset_id', sa.Integer(), nullable=False),
        sa.Column('base_version_id', sa.Integer(), nullable=True),
        sa.Column('created_version_id', sa.Integer(), nullable=True),
        sa.Column(
            'status',
            postgresql.ENUM(name='preprocessingstatusenum', create_type=False),
            nullable=False,
        ),
        sa.Column('result', autoupdateresult_enum, nullable=True),
        sa.Column('details', sa.String(), nullable=True),
        sa.Column('reason_for_failure', sa.String(), nullable=True),
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
        sa.ForeignKeyConstraint(['base_version_id'], ['channel_dataset_versions.id']),
        sa.ForeignKeyConstraint(['channel_dataset_id'], ['channel_datasets.id']),
        sa.ForeignKeyConstraint(['created_version_id'], ['channel_dataset_versions.id']),
        sa.PrimaryKeyConstraint('id'),
    )

    op.create_index(
        'ix_auto_update_jobs_channel_dataset_id',
        'auto_update_jobs',
        ['channel_dataset_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_auto_update_jobs_channel_dataset_id', table_name='auto_update_jobs')
    op.drop_table('auto_update_jobs')
    op.execute("DROP TYPE IF EXISTS autoupdateresult")

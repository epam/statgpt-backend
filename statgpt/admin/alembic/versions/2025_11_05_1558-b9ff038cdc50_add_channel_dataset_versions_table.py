"""Add `channel_dataset_versions` table

Revision ID: b9ff038cdc50
Revises: 04eca13e7368
Create Date: 2025-11-05 15:58:11.308348

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'b9ff038cdc50'
down_revision: str | None = '04eca13e7368'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'channel_dataset_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('channel_dataset_id', sa.Integer(), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column(
            'preprocessing_status',
            postgresql.ENUM(name='preprocessingstatusenum', create_type=False),
            nullable=False,
        ),
        sa.Column('creation_reason', sa.String(), nullable=False),
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
        sa.ForeignKeyConstraint(
            ['channel_dataset_id'],
            ['channel_datasets.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'channel_dataset_id', 'version', name='uix_unique_version_for_channel_dataset'
        ),
    )

    # ~~~~~ Create trigger function to auto-increment version ~~~~~

    op.execute(
        """
        CREATE OR REPLACE FUNCTION increment_channel_dataset_version()
        RETURNS TRIGGER AS $$
        BEGIN
            -- Check if version is NULL or 0 (the dummy default)
            IF NEW.version IS NULL OR NEW.version = 0 THEN
                SELECT COALESCE(MAX(version), 0) + 1
                INTO NEW.version
                FROM channel_dataset_versions
                WHERE channel_dataset_id = NEW.channel_dataset_id;
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # Create the trigger
    op.execute(
        """
        CREATE TRIGGER set_channel_dataset_version
            BEFORE INSERT
            ON channel_dataset_versions
            FOR EACH ROW
        EXECUTE FUNCTION increment_channel_dataset_version();
        """
    )

    # Create index for performance
    op.create_index(
        'idx_channel_dataset_version',
        'channel_dataset_versions',
        ['channel_dataset_id', 'version'],
        postgresql_using='btree',
    )

    # ~~~~~ Drop previous data schema ~~~~~

    op.drop_column('channel_datasets', 'preprocessing_status')


def downgrade() -> None:
    op.add_column(
        'channel_datasets',
        sa.Column(
            'preprocessing_status',
            postgresql.ENUM(name='preprocessingstatusenum', create_type=False),
            server_default=sa.text("'NOT_STARTED'::preprocessingstatusenum"),
            autoincrement=False,
            nullable=False,
        ),
    )

    op.execute("DROP TRIGGER IF EXISTS set_channel_dataset_version ON channel_dataset_versions")
    op.execute("DROP FUNCTION IF EXISTS increment_channel_dataset_version()")
    op.drop_index('idx_channel_dataset_version', 'channel_dataset_versions')

    op.drop_table('channel_dataset_versions')

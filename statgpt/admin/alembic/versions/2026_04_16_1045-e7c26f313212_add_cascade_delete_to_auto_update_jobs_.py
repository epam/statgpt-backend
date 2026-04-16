"""add cascade delete to auto_update_jobs channel_dataset FK

Revision ID: e7c26f313212
Revises: 2f0f6f2f0d7b
Create Date: 2026-04-16 10:45:58.004322

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e7c26f313212'
down_revision: str | None = '2f0f6f2f0d7b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        'auto_update_jobs_channel_dataset_id_fkey', 'auto_update_jobs', type_='foreignkey'
    )
    op.create_foreign_key(
        'auto_update_jobs_channel_dataset_id_fkey',
        'auto_update_jobs',
        'channel_datasets',
        ['channel_dataset_id'],
        ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    op.drop_constraint(
        'auto_update_jobs_channel_dataset_id_fkey', 'auto_update_jobs', type_='foreignkey'
    )
    op.create_foreign_key(
        'auto_update_jobs_channel_dataset_id_fkey',
        'auto_update_jobs',
        'channel_datasets',
        ['channel_dataset_id'],
        ['id'],
    )

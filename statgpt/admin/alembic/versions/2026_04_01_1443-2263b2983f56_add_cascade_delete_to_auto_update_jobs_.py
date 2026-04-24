"""add cascade delete to auto_update_jobs version FKs

Revision ID: 2263b2983f56
Revises: 75cf43005ed1
Create Date: 2026-04-01 14:43:53.221299

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '2263b2983f56'
down_revision: str | None = '75cf43005ed1'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        'auto_update_jobs_base_version_id_fkey', 'auto_update_jobs', type_='foreignkey'
    )
    op.drop_constraint(
        'auto_update_jobs_created_version_id_fkey', 'auto_update_jobs', type_='foreignkey'
    )
    op.create_foreign_key(
        'auto_update_jobs_base_version_id_fkey',
        'auto_update_jobs',
        'channel_dataset_versions',
        ['base_version_id'],
        ['id'],
        ondelete='CASCADE',
    )
    op.create_foreign_key(
        'auto_update_jobs_created_version_id_fkey',
        'auto_update_jobs',
        'channel_dataset_versions',
        ['created_version_id'],
        ['id'],
        ondelete='CASCADE',
    )


def downgrade() -> None:
    op.drop_constraint(
        'auto_update_jobs_created_version_id_fkey', 'auto_update_jobs', type_='foreignkey'
    )
    op.drop_constraint(
        'auto_update_jobs_base_version_id_fkey', 'auto_update_jobs', type_='foreignkey'
    )
    op.create_foreign_key(
        'auto_update_jobs_base_version_id_fkey',
        'auto_update_jobs',
        'channel_dataset_versions',
        ['base_version_id'],
        ['id'],
    )
    op.create_foreign_key(
        'auto_update_jobs_created_version_id_fkey',
        'auto_update_jobs',
        'channel_dataset_versions',
        ['created_version_id'],
        ['id'],
    )

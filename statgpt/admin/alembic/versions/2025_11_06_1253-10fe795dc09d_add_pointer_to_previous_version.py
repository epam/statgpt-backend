"""Add pointer to previous version

Revision ID: 10fe795dc09d
Revises: b9ff038cdc50
Create Date: 2025-11-06 12:53:32.235349

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '10fe795dc09d'
down_revision: str | None = 'b9ff038cdc50'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


CONSTRAINT_NAME = 'channel_dataset_versions_pointer_to_fkey'


def upgrade() -> None:
    op.add_column('channel_dataset_versions', sa.Column('pointer_to', sa.Integer(), nullable=True))
    op.create_foreign_key(
        CONSTRAINT_NAME,
        'channel_dataset_versions',
        'channel_dataset_versions',
        ['pointer_to'],
        ['id'],
        ondelete='SET NULL',
    )


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT_NAME, 'channel_dataset_versions', type_='foreignkey')
    op.drop_column('channel_dataset_versions', 'pointer_to')

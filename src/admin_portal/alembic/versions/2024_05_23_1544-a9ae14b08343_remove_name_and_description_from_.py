"""Remove name and description from datasets table

Revision ID: a9ae14b08343
Revises: ec74743d00c7
Create Date: 2024-05-23 15:44:52.869237

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a9ae14b08343'
down_revision: Union[str, None] = 'ec74743d00c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('datasets', 'description')
    op.drop_column('datasets', 'title')


def downgrade() -> None:
    op.add_column(
        'datasets',
        sa.Column('title', sa.VARCHAR(), server_default="", autoincrement=False, nullable=False),
    )
    op.add_column(
        'datasets',
        sa.Column(
            'description',
            sa.VARCHAR(),
            server_default=sa.text("''::character varying"),
            autoincrement=False,
            nullable=False,
        ),
    )

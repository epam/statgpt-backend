"""Add glossary term model

Revision ID: 20072b7c9bf3
Revises: c5faf155dafe
Create Date: 2025-02-26 17:05:59.898249

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '20072b7c9bf3'
down_revision: Union[str, None] = 'c5faf155dafe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'glossary_terms',
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column('term', sa.String(), nullable=False),
        sa.Column('definition', sa.String(), nullable=False),
        sa.Column('domain', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
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
            ['channel_id'],
            ['channels.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('glossary_terms')

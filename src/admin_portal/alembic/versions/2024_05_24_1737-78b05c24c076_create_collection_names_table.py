"""Create 'collection_names' table

Revision ID: 78b05c24c076
Revises: a9ae14b08343
Create Date: 2024-05-24 17:37:02.301868

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '78b05c24c076'
down_revision: Union[str, None] = 'a9ae14b08343'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS collections")
    op.create_table(
        '_names',
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('collection_name', sa.String(), nullable=False),
        sa.Column('datasource', sa.String(), nullable=True, server_default=None),
        sa.Column('embedding_model_name', sa.String(), nullable=False),
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
        sa.PrimaryKeyConstraint('uuid'),
        schema='collections',
    )


def downgrade() -> None:
    op.drop_table('_names', schema='collections')

"""Add jobs table

Revision ID: dd4eca09ec3f
Revises: 1981d81ad692
Create Date: 2024-10-29 16:41:08.237132

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'dd4eca09ec3f'
down_revision: Union[str, None] = '1981d81ad692'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('type', sa.Enum('EXPORT', 'IMPORT', name='jobtype'), nullable=False),
        sa.Column(
            'status',
            postgresql.ENUM(name='preprocessingstatusenum', create_type=False),
            nullable=False,
        ),
        sa.Column('file', sa.String(), nullable=True),
        sa.Column('channel_id', sa.Integer(), nullable=True),
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
            ['channel_id'],
            ['channels.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )


def downgrade() -> None:
    op.drop_table('jobs')
    op.execute("DROP TYPE jobtype;")

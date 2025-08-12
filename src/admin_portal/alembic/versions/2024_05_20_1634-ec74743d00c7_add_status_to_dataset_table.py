"""Add status to dataset table

Revision ID: ec74743d00c7
Revises: 9516b91426b5
Create Date: 2024-05-20 16:34:20.935766

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'ec74743d00c7'
down_revision: Union[str, None] = '9516b91426b5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'datasets',
        sa.Column(
            'preprocessing_status',
            sa.Enum(
                'NOT_STARTED', 'IN_PROGRESS', 'COMPLETED', 'FAILED', name='preprocessingstatusenum'
            ),
            nullable=False,
            server_default='COMPLETED',
        ),
    )


def downgrade() -> None:
    op.drop_column('datasets', 'preprocessing_status')

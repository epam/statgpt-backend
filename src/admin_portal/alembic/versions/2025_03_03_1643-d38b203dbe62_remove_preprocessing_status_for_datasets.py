"""Remove field `preprocessing_status` from table `datasets`

Revision ID: d38b203dbe62
Revises: 20072b7c9bf3
Create Date: 2025-03-03 16:43:01.931274

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd38b203dbe62'
down_revision: Union[str, None] = '20072b7c9bf3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column('datasets', 'preprocessing_status')


def downgrade() -> None:
    op.add_column(
        'datasets',
        sa.Column(
            'preprocessing_status',
            postgresql.ENUM(
                'NOT_STARTED',
                'IN_PROGRESS',
                'COMPLETED',
                'FAILED',
                'QUEUED',
                name='preprocessingstatusenum',
            ),
            server_default=sa.text("'COMPLETED'::preprocessingstatusenum"),
            autoincrement=False,
            nullable=False,
        ),
    )

"""Add new status

Revision ID: 7d5f221575d7
Revises: ab73a3a7de72
Create Date: 2024-05-31 14:50:52.115891

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '7d5f221575d7'
down_revision: Union[str, None] = 'ab73a3a7de72'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE preprocessingstatusenum ADD VALUE 'QUEUED'")


def downgrade() -> None:
    pass

"""Update Enum

Revision ID: 9516b91426b5
Revises: 26d45d144edd
Create Date: 2024-05-20 15:02:56.296536

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '9516b91426b5'
down_revision: Union[str, None] = '26d45d144edd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TYPE preprocessingstatusenum ADD VALUE 'FAILED'")


def downgrade() -> None:
    pass

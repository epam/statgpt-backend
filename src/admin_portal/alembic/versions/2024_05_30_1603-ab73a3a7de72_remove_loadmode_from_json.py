"""Remove LoadMode from JSON

Revision ID: ab73a3a7de72
Revises: 9c86e3f19f2c
Create Date: 2024-05-30 16:03:03.281996

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'ab73a3a7de72'
down_revision: Union[str, None] = '9c86e3f19f2c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE datasets SET details = details::jsonb #- '{loadMode}';")


def downgrade() -> None:
    op.execute("UPDATE datasets SET details = details::jsonb || jsonb '{\"loadMode\":\"VERIFY\"}';")

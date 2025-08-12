"""Add DatasetCitation

Revision ID: 1981d81ad692
Revises: 7d5f221575d7
Create Date: 2024-06-05 11:28:51.607496

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '1981d81ad692'
down_revision: Union[str, None] = '7d5f221575d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE datasets SET details = details::jsonb || jsonb '{\"citation\": null}';")


def downgrade() -> None:
    op.execute("UPDATE datasets SET details = details::jsonb #- '{citation}';")

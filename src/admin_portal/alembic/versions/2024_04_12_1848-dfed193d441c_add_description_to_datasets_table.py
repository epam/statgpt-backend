"""Add description to datasets table

Revision ID: dfed193d441c
Revises: bee4454de35e
Create Date: 2024-04-12 18:48:35.920770

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dfed193d441c"
down_revision: Union[str, None] = "bee4454de35e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "datasets", sa.Column("description", sa.String(), server_default="", nullable=False)
    )


def downgrade() -> None:
    op.drop_column("datasets", "description")

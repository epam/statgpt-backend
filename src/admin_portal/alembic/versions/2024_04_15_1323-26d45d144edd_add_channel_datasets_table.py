"""Add channel_datasets table

Revision ID: 26d45d144edd
Revises: dfed193d441c
Create Date: 2024-04-15 13:23:22.061168

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "26d45d144edd"
down_revision: Union[str, None] = "dfed193d441c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP TYPE IF EXISTS preprocessingstatusenum;")

    op.create_table(
        "channel_datasets",
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column(
            "preprocessing_status",
            sa.Enum("NOT_STARTED", "IN_PROGRESS", "COMPLETED", name="preprocessingstatusenum"),
            nullable=False,
        ),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["channel_id"],
            ["channels.id"],
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("channel_datasets")
    op.execute("DROP TYPE preprocessingstatusenum;")

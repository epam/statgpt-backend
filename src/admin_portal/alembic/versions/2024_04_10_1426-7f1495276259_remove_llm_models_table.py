"""Remove llm_models table

Revision ID: 7f1495276259
Revises: be6a6d78fb51
Create Date: 2024-04-10 14:26:26.508051

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "7f1495276259"
down_revision: Union[str, None] = "be6a6d78fb51"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "channels",
        sa.Column(
            "llm_model", sa.String(), nullable=False, server_default="text-embedding-3-large"
        ),
    )
    op.drop_constraint("channels_llm_model_id_fkey", "channels", type_="foreignkey")
    op.drop_column("channels", "llm_model_id")
    op.drop_table("llm_models")


def downgrade() -> None:
    op.create_table(
        "llm_models",
        sa.Column("name", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("description", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            postgresql.TIMESTAMP(timezone=True),
            server_default=sa.text("now()"),
            autoincrement=False,
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="llm_models_pkey"),
    )

    op.execute("INSERT INTO llm_models(name, description) VALUES ('text-embedding-3-large', '');")

    op.drop_column("channels", "llm_model")
    op.add_column(
        "channels",
        sa.Column(
            "llm_model_id", sa.INTEGER(), autoincrement=False, nullable=False, server_default="1"
        ),
    )
    op.create_foreign_key(
        "channels_llm_model_id_fkey", "channels", "llm_models", ["llm_model_id"], ["id"]
    )

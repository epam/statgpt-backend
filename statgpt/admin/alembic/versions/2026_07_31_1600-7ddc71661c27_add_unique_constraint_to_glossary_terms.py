"""add unique constraint to glossary_terms

Revision ID: 7ddc71661c27
Revises: 85c92803e84c
Create Date: 2026-07-31 16:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '7ddc71661c27'
down_revision: str | None = '85c92803e84c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Some environments already contain duplicate glossary terms (same channel_id
    # and term) created by earlier imports. Remove them, keeping the most recently
    # inserted row per (channel_id, term), so the unique constraint can be added.
    # This DELETE is irreversible, so report how many rows were dropped to give the
    # operator a record (e.g. a hand-curated `GDP` kept in two domains).
    result = op.get_bind().execute(sa.text("""
            DELETE FROM glossary_terms
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM glossary_terms
                GROUP BY channel_id, term
            )
            """))
    if result.rowcount:
        print(
            f"Removed {result.rowcount} duplicate glossary term(s) "
            "before adding uq_glossary_terms_channel_id_term."
        )
    op.create_unique_constraint(
        'uq_glossary_terms_channel_id_term',
        'glossary_terms',
        ['channel_id', 'term'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_glossary_terms_channel_id_term',
        'glossary_terms',
        type_='unique',
    )

"""add unique constraint to glossary_terms

Revision ID: 7ddc71661c27
Revises: 85c92803e84c
Create Date: 2026-07-31 16:00:00.000000

"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = '7ddc71661c27'
down_revision: str | None = '85c92803e84c'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Some environments already contain duplicate glossary terms (same channel_id
    # and term) created by earlier imports. Remove them, keeping the most recently
    # inserted row per (channel_id, term), so the unique constraint can be added.
    # `downgrade()` cannot restore these rows, so log every one that is dropped: it is
    # the operator's only record of e.g. a hand-curated `GDP` kept in two domains.
    deleted = op.get_bind().execute(sa.text("""
            DELETE FROM glossary_terms
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM glossary_terms
                GROUP BY channel_id, term
            )
            RETURNING channel_id, term
            """)).all()
    if deleted:
        dropped = ", ".join(f"channel_id={row.channel_id} term={row.term!r}" for row in deleted)
        logger.warning(
            f"Removed {len(deleted)} duplicate glossary term(s) before adding "
            f"uq_glossary_terms_channel_id_term: {dropped}"
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

"""separate indicator sheets for grade c discovery

Revision ID: d7339387d7fc
Revises: a4f1d794d575
Create Date: 2026-09-16 10:42:48.713887

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd7339387d7fc'
down_revision: str | None = 'a4f1d794d575'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Added with a server default so the existing rows get a value, then dropped again: the
    # column is written by the application, and a default left in the schema would let a row
    # be inserted without one.
    op.add_column(
        'discovery_datasets',
        sa.Column('indicators_summary', sa.String(), nullable=False, server_default=''),
    )
    op.alter_column('discovery_datasets', 'indicators_summary', server_default=None)

    op.drop_column('discovery_datasets', 'excluded_regional_values')

    # `indicators_summary` is deliberately left empty. The only thing that publishes is an
    # indexing run, and the run derives the column before its publish stage, so an existing
    # record's first run copies its stored `indicators_coverage` across and its document
    # carries the same value it carries today.
    #
    # `indicators_coverage` needs no migration either: what the old template collected is
    # already the ';'-joined form an indicators sheet now produces. `missing_indicators`
    # keeps the submitted values until a run derives them.

    # Every verdict and every published document describes the record under the old schema:
    # the documents carry metadata this schema no longer declares, and no record has been
    # judged against the new indicator-list check. So derived state is reset and an admin
    # reindexes, rather than the index being left serving documents nothing here can explain.
    op.execute("""
        UPDATE discovery_datasets
        SET validation_status = 'NOT_VALIDATED',
            validation_issues = NULL,
            validated_at = NULL,
            indexing_status = CASE indexing_status
                WHEN 'INDEXED' THEN 'OUTDATED'
                WHEN 'FAILED' THEN 'NEW'
                ELSE indexing_status
            END,
            index_error = CASE indexing_status WHEN 'FAILED' THEN NULL ELSE index_error END
        """)


def downgrade() -> None:
    op.add_column(
        'discovery_datasets',
        sa.Column('excluded_regional_values', sa.String(), nullable=False, server_default=''),
    )
    op.alter_column('discovery_datasets', 'excluded_regional_values', server_default=None)
    op.drop_column('discovery_datasets', 'indicators_summary')

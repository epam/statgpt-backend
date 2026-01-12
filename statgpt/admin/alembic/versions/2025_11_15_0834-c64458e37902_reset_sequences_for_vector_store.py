"""Reset sequences for vector store

Revision ID: c64458e37902
Revises: d528d881ece8
Create Date: 2025-11-15 08:34:48.017064

"""

import logging
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = 'c64458e37902'
down_revision: str | None = 'd528d881ece8'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Reset sequences for vector store tables to prevent duplicate key errors.

    This migration fixes sequence values for tables that had explicit IDs set during import,
    which caused the sequences to fall behind the actual maximum ID values in the tables.
    """
    conn = op.get_bind()

    # Get all tables in the collections schema matching the prefixes
    result = conn.execute(
        sa.text(
            """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'collections'
          AND table_type = 'BASE TABLE'
          AND (table_name LIKE 'AvailableDimensions%'
               OR table_name LIKE 'Indicators%'
               OR table_name LIKE 'SpecialDimensions%')
        ORDER BY table_name
    """
        )
    )

    tables = [row[0] for row in result]

    for table_name in tables:
        # Get the sequence name for the id column
        seq_result = conn.execute(
            sa.text(
                """
            SELECT pg_get_serial_sequence(:full_table_name, 'id')
        """
            ),
            {"full_table_name": f'collections."{table_name}"'},
        )

        sequence_name = seq_result.scalar()

        if sequence_name:
            # Reset the sequence to MAX(id) + 1
            # Use COALESCE to handle empty tables (set to 1 in that case)
            conn.execute(
                sa.text(
                    f"""
                SELECT setval(
                    :sequence_name,
                    COALESCE((SELECT MAX(id) FROM collections."{table_name}"), 1)
                )
            """
                ),
                {"sequence_name": sequence_name},
            )

            logger.info(f"Reset sequence for table: {table_name} -> {sequence_name}")
        else:
            logger.info(f"No sequence found for table: {table_name}")


def downgrade() -> None:
    # This migration cannot be reversed as we don't know the original sequence values
    pass

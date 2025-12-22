"""Drop outdated collections tables

Revision ID: f006354afe36
Revises: 5fac4226f0af
Create Date: 2025-12-19 14:52:07.783585

This migration removes legacy tables that were generated dynamically per-environment:
- collections._names
- collections.c_<uuid>
- collections.c_<uuid>_mapping

UUIDs vary per environment, so the upgrade discovers and drops matching tables at runtime.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f006354afe36'
down_revision: str | None = '5fac4226f0af'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
DO $$
DECLARE
  r RECORD;
BEGIN
  FOR r IN
    SELECT schemaname, tablename
    FROM pg_tables
    WHERE schemaname = 'collections'
      AND (
        tablename = '_names'
        OR tablename ~ '^c_[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        OR tablename ~ '^c_[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}_mapping$'
      )
  LOOP
    EXECUTE format('DROP TABLE IF EXISTS %I.%I CASCADE', r.schemaname, r.tablename);
  END LOOP;
END $$;
"""
    )


def downgrade() -> None:
    # Only "_names" table can be restored to the previous state.
    op.execute("CREATE SCHEMA IF NOT EXISTS collections")
    op.execute(
        """
CREATE TABLE IF NOT EXISTS collections."_names" (
  uuid UUID PRIMARY KEY,
  collection_name VARCHAR NOT NULL,
  datasource VARCHAR NULL,
  embedding_model_name VARCHAR NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
    )

"""Add scope column to audit logs

Revision ID: f4c2a9b7d3e1
Revises: a4f1d794d575
Create Date: 2026-08-14 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'f4c2a9b7d3e1'
down_revision: str | None = 'a4f1d794d575'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    audit_scope = postgresql.ENUM(
        'config',
        'ex_im',
        'reindex',
        'ds_link',
        name='auditscope',
        create_type=False,
    )
    audit_scope.create(op.get_bind(), checkfirst=True)

    op.add_column(
        'audit_logs',
        sa.Column('scope', audit_scope, nullable=False, server_default='config'),
    )

    # Backfill historical export/import records. audit_logs rows are immutable
    # via a trigger, so it must be toggled off for this one-time backfill.
    op.execute("ALTER TABLE audit_logs DISABLE TRIGGER trg_prevent_audit_log_mutation")
    op.execute("UPDATE audit_logs SET scope = 'ex_im' WHERE entity_type = 'import_job'")
    op.execute("ALTER TABLE audit_logs ENABLE TRIGGER trg_prevent_audit_log_mutation")


def downgrade() -> None:
    op.drop_column('audit_logs', 'scope')
    postgresql.ENUM(name='auditscope').drop(op.get_bind(), checkfirst=True)

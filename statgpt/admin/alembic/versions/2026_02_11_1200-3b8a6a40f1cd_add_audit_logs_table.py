"""Add immutable audit logs table

Revision ID: 3b8a6a40f1cd
Revises: c7f068b2d47d
Create Date: 2026-02-11 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3b8a6a40f1cd'
down_revision: str | None = 'c7f068b2d47d'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    postgresql.ENUM(
        'EXISTS',
        'ABSENT',
        'CREATED',
        'MODIFIED',
        'DELETED',
        'NOT_CHANGED',
        name='auditstateenum',
    ).create(op.get_bind(), checkfirst=True)

    audit_state_enum = postgresql.ENUM(
        'EXISTS',
        'ABSENT',
        'CREATED',
        'MODIFIED',
        'DELETED',
        'NOT_CHANGED',
        name='auditstateenum',
        create_type=False,
    )

    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('entity_type', sa.String(), nullable=False),
        sa.Column('action_type', sa.String(), nullable=False),
        sa.Column('entity_id', sa.String(), nullable=True),
        sa.Column('entity_name', sa.String(), nullable=True),
        sa.Column('performed_by', sa.String(), nullable=True),
        sa.Column('performed_by_name', sa.String(), nullable=True),
        sa.Column('action_trigger', sa.String(), nullable=False),
        sa.Column('state_before', audit_state_enum, nullable=False),
        sa.Column('state_after', audit_state_enum, nullable=False),
        sa.Column('trace_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_audit_logs_created_at', 'audit_logs', ['created_at'], unique=False)
    op.create_index(
        'ix_audit_logs_entity_type_entity_id',
        'audit_logs',
        ['entity_type', 'entity_id'],
        unique=False,
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION prevent_audit_log_mutation()
        RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_logs rows are immutable';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_prevent_audit_log_mutation
        BEFORE UPDATE OR DELETE ON audit_logs
        FOR EACH ROW
        EXECUTE FUNCTION prevent_audit_log_mutation();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_prevent_audit_log_mutation ON audit_logs")
    op.execute("DROP FUNCTION IF EXISTS prevent_audit_log_mutation()")
    op.drop_index('ix_audit_logs_entity_type_entity_id', table_name='audit_logs')
    op.drop_index('ix_audit_logs_created_at', table_name='audit_logs')
    op.drop_table('audit_logs')
    postgresql.ENUM(name='auditstateenum').drop(op.get_bind(), checkfirst=True)

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
    audit_entity_type = postgresql.ENUM(
        'channel',
        'dataset',
        'data_source',
        'import_job',
        name='auditentitytype',
        create_type=False,
    )
    audit_action_type = postgresql.ENUM(
        'create',
        'update',
        'delete',
        name='auditactiontype',
        create_type=False,
    )
    audit_entity_type.create(op.get_bind(), checkfirst=True)
    audit_action_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('entity_type', audit_entity_type, nullable=False),
        sa.Column('action_type', audit_action_type, nullable=False),
        sa.Column('item_id', sa.Integer(), nullable=True),
        sa.Column('entity_id', sa.String(), nullable=True),
        sa.Column('entity_name', sa.String(), nullable=True),
        sa.Column('performed_by', sa.String(), nullable=True),
        sa.Column('performed_by_name', sa.String(), nullable=True),
        sa.Column('state_after', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
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
    op.create_index(
        'ix_audit_logs_entity_type_item_id',
        'audit_logs',
        ['entity_type', 'item_id'],
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
    op.drop_index('ix_audit_logs_entity_type_item_id', table_name='audit_logs')
    op.drop_index('ix_audit_logs_entity_type_entity_id', table_name='audit_logs')
    op.drop_index('ix_audit_logs_created_at', table_name='audit_logs')
    op.drop_table('audit_logs')
    postgresql.ENUM(name='auditactiontype').drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name='auditentitytype').drop(op.get_bind(), checkfirst=True)

"""Initial version

Revision ID: fa6b0a97e522
Revises: None (initial version)
Create Date: 2025-09-21 14:12:15.114492

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'fa6b0a97e522'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_collections_schema() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS collections")
    op.create_table(
        '_names',
        sa.Column('uuid', postgresql.UUID(), nullable=False),
        sa.Column('collection_name', sa.String(), nullable=False),
        sa.Column('datasource', sa.String(), nullable=True, server_default=None),
        sa.Column('embedding_model_name', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('uuid'),
        schema='collections',
    )


def _drop_collections_schema() -> None:
    op.drop_table('_names', schema='collections')
    op.execute("DROP SCHEMA IF EXISTS collections")


def _create_public_schema() -> None:
    op.create_table(
        'channels',
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('deployment_id', sa.String(), nullable=False),
        sa.Column('llm_model', sa.String(), nullable=False),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('deployment_id'),
    )
    op.create_table(
        'data_source_types',
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'data_sources',
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=False),
        sa.Column('type_id', sa.Integer(), nullable=False),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['type_id'],
            ['data_source_types.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'glossary_terms',
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column('term', sa.String(), nullable=False),
        sa.Column('definition', sa.String(), nullable=False),
        sa.Column('domain', sa.String(), nullable=False),
        sa.Column('source', sa.String(), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['channel_id'],
            ['channels.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'jobs',
        sa.Column('type', sa.Enum('EXPORT', 'IMPORT', name='jobtype'), nullable=False),
        sa.Column(
            'status',
            sa.Enum(
                'NOT_STARTED',
                'QUEUED',
                'IN_PROGRESS',
                'COMPLETED',
                'FAILED',
                name='preprocessingstatusenum',
            ),
            nullable=False,
        ),
        sa.Column('file', sa.String(), nullable=True),
        sa.Column('channel_id', sa.Integer(), nullable=True),
        sa.Column('reason_for_failure', sa.String(), nullable=True),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['channel_id'],
            ['channels.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_table(
        'datasets',
        sa.Column('id_', sa.UUID(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['source_id'],
            ['data_sources.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('id_'),
    )

    op.create_table(
        'channel_datasets',
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),
        sa.Column(
            'preprocessing_status',
            sa.Enum(
                'NOT_STARTED',
                'QUEUED',
                'IN_PROGRESS',
                'COMPLETED',
                'FAILED',
                name='preprocessingstatusenum',
            ),
            nullable=False,
        ),
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['channel_id'],
            ['channels.id'],
        ),
        sa.ForeignKeyConstraint(
            ['dataset_id'],
            ['datasets.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )


def _drop_public_schema() -> None:
    op.drop_table('channel_datasets')
    op.drop_table('datasets')
    op.drop_table('jobs')
    op.drop_table('glossary_terms')
    op.drop_table('data_sources')
    op.drop_table('data_source_types')
    op.drop_table('channels')
    op.execute("DROP TYPE IF EXISTS jobtype")
    op.execute("DROP TYPE IF EXISTS preprocessingstatusenum")


def upgrade() -> None:
    _create_collections_schema()
    _create_public_schema()


def downgrade() -> None:
    _drop_collections_schema()
    _drop_public_schema()

"""Add uuid and title to `datasets` table

Revision ID: 318707f46c84
Revises: d38b203dbe62
Create Date: 2025-04-17 14:36:11.806280

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '318707f46c84'
down_revision: Union[str, None] = 'd38b203dbe62'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


title_to_uuid = {
    'Advanced Insurance Data': 'b3d46d99-d777-444a-8d44-962997e3e757',
    'BIS, Debt securities statistics': '72a9a53d-39b3-4ddb-9b71-6d7e83c3c9e1',
    'ECB, Balance Sheet Items': '166a384f-795d-4b4c-ae5a-381d4bbd1337',
    'ESTAT, GDP and main components (output, expenditure and income)': '218d4c14-ed12-4cde-901e-13c9b5d39597',
    'FRED, Household Debt Service and Financial Obligations Ratios (FOR)': '924a0db7-95d8-4b77-bff1-33502af0621c',
    'IMF, World Economic Outlook (WEO)': '68127d32-e106-47b6-a0a2-14b6fa5253c1',
    'Insurance Protection Gap Data': 'be9b6912-1c4c-4efe-a28b-1d52dee5012a',
    'Macro Scenarios': '5fb0219a-c6e1-458e-a45f-6d339b62882b',
    'Natural Catastrophe Data': 'd70ff7c9-b0e2-4d63-a68e-c22b7b9e9ffc',
    'OECD, Trade in Value Added (TiVA) 2023 edition: Principal Indicators': 'a1c08f25-bee9-4a62-85b0-97367e3699a6',
    'Proprietary Indexes': 'e7d700b3-d049-42ee-b690-b10d081db51c',
    'Standard Economics Data': 'dc087aa2-9614-49de-9ea8-bd596718da67',
    'Standard Insurance Data': '555cfa2a-1de7-412d-b190-9ca51ff9d8ab',
    'StatCan, National Balance Sheet Accounts': '0fcdf4ca-89d7-496e-ae40-fb8800795c0a',
    'WB, World Development Indicators (WDI)': '75187702-2b0d-4fba-8c87-ab26c9544628',
}


def upgrade() -> None:
    op.add_column('datasets', sa.Column('id_', sa.UUID(), nullable=True))
    op.add_column('datasets', sa.Column('title', sa.String(), nullable=True))

    # Update title from details.titleOverride
    op.execute(
        """
        UPDATE datasets
        SET title = details->>'titleOverride'
        """
    )

    # Update `id_` based on `title_to_uuid` mapping
    for title, uuid_value in title_to_uuid.items():
        op.execute(
            f"""
            UPDATE datasets
            SET id_ = '{uuid_value}'
            WHERE title = '{title}'
            """
        )

    # For remaining rows without `id_`, generate new UUIDs
    op.execute(
        """
        UPDATE datasets
        SET id_ = gen_random_uuid()
        WHERE id_ IS NULL
        """
    )

    op.alter_column('datasets', 'id_', existing_type=sa.UUID(), nullable=False)
    op.alter_column('datasets', 'title', existing_type=sa.String(), nullable=False)
    op.create_unique_constraint('datasets_unique_uuid', 'datasets', ['id_'])


def downgrade() -> None:
    # Update or add titleOverride in the `details` field
    for title, uuid_value in title_to_uuid.items():
        op.execute(
            f"""
            UPDATE datasets
            SET details = details || '{{"titleOverride": "{title}"}}'::jsonb
            WHERE id_ = '{uuid_value}'
            """
        )

    op.drop_constraint('datasets_unique_uuid', 'datasets', type_='unique')
    op.drop_column('datasets', 'title')
    op.drop_column('datasets', 'id_')

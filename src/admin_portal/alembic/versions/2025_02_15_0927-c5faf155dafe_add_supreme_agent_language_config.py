"""add supreme agent language config

Revision ID: c5faf155dafe
Revises: 6434c19ab7a9
Create Date: 2025-02-15 09:27:09.915409

"""

import json
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c5faf155dafe'
down_revision: Union[str, None] = '6434c19ab7a9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    data = {
        "terminologyDomain": "",
        "languageInstructions": [],
    }

    op.execute(
        f"""
        UPDATE channels
        SET details = jsonb_set(
            details::jsonb,
            '{{supremeAgent}}',
            (details->'supremeAgent')::jsonb || '{json.dumps(data)}'::jsonb
        );
        """
    )


def downgrade() -> None:
    # Remove only supremeAgent->terminologyDomain and supremeAgent->languageInstructions from the details
    op.execute(
        """
        UPDATE channels
        SET details = details::jsonb
            #- '{supremeAgent,terminologyDomain}'
            #- '{supremeAgent,languageInstructions}';
        """
    )

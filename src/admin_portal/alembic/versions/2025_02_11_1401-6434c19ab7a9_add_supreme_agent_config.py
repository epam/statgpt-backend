"""Add supreme agent config

Revision ID: 6434c19ab7a9
Revises: dd4eca09ec3f
Create Date: 2025-02-11 14:01:32.597549

"""

import json
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '6434c19ab7a9'
down_revision: Union[str, None] = 'dd4eca09ec3f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    data = {
        "supremeAgent": {
            "name": "StatGPT",
            "domain": "Statistics, economics and SDMX.",
        }
    }

    op.execute(f"UPDATE channels SET details = details::jsonb || jsonb '{json.dumps(data)}';")

    swre_data = {
        "supremeAgent": {
            "name": "Swiss Re AI Assistant",
            "domain": (
                "Insurance market, economics and economic policies,"
                " impact of catastrophes and statistical data related to these domains."
            ),
        }
    }
    op.execute(
        f"UPDATE channels"
        f" SET details = details::jsonb || jsonb '{json.dumps(swre_data)}'"
        f" WHERE deployment_id LIKE '%swre%';"
    )


def downgrade() -> None:
    op.execute("UPDATE channels SET details = details::jsonb #- '{supremeAgent}';")

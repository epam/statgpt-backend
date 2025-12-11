"""Update AuthConfig config

Revision ID: 65142518fe0c
Revises: 65c149c7db9e
Create Date: 2025-12-03 11:32:05.932867

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '65142518fe0c'
down_revision: str | None = '65c149c7db9e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Rename grantType to systemAuthType in AuthConfig for QH_SDMX21 data sources
    op.execute(
        """
        UPDATE data_sources
        SET details = jsonb_set(
            details #- '{authConfig,grantType}',
            '{authConfig,systemAuthType}',
            details #> '{authConfig,grantType}'
        )
        WHERE type_id = (SELECT id FROM data_source_types WHERE name = 'QH_SDMX21')
        AND details -> 'authConfig' IS NOT NULL
        AND details -> 'authConfig' ? 'grantType'
    """
    )


def downgrade() -> None:
    # Rename systemAuthType back to grantType in AuthConfig for QH_SDMX21 data sources
    op.execute(
        """
        UPDATE data_sources
        SET details = jsonb_set(
            details #- '{authConfig,systemAuthType}',
            '{authConfig,grantType}',
            details #> '{authConfig,systemAuthType}'
        )
        WHERE type_id = (SELECT id FROM data_source_types WHERE name = 'QH_SDMX21')
        AND details -> 'authConfig' IS NOT NULL
        AND details -> 'authConfig' ? 'systemAuthType'
    """
    )

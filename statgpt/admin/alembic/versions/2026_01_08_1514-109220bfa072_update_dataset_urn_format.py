"""Update dataset URN format

Revision ID: 109220bfa072
Revises: f006354afe36
Create Date: 2026-01-08 15:14:00.000000

"""

import json
import logging
import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = '109220bfa072'
down_revision: str | None = 'f006354afe36'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def parse_short_urn(urn_string: str) -> dict[str, str]:
    """Parse short URN format 'agency:resource(version)' into components."""
    # Pattern: agency:resource(version)
    pattern = r'^([^:]+):([^(]+)\(([^)]+)\)$'
    match = re.match(pattern, urn_string)

    if not match:
        raise ValueError(f"Invalid URN format: {urn_string}")

    agency_id, resource_id, version = match.groups()
    return {"agency_id": agency_id, "resource_id": resource_id, "version": version}


def upgrade() -> None:
    connection = op.get_bind()

    # Fetch all datasets
    result = connection.execute(sa.text("SELECT id, details FROM datasets"))

    for row in result:
        dataset_id = row[0]
        details = row[1]

        # Check if 'urn' field exists and is a string
        if 'urn' in details and isinstance(details['urn'], str):
            try:
                # Parse the old URN format
                urn_obj = parse_short_urn(details['urn'])

                # Update the details with the new URN object structure
                details['urn'] = urn_obj

                # Update the database - serialize dict to JSON string for JSONB column
                connection.execute(
                    sa.text("UPDATE datasets SET details = CAST(:details AS jsonb) WHERE id = :id"),
                    {"details": json.dumps(details), "id": dataset_id},
                )
            except ValueError as e:
                # Log or handle parsing errors gracefully
                logger.warning(f"Failed to parse URN for dataset {dataset_id}: {e}")


def downgrade() -> None:
    connection = op.get_bind()

    # Fetch all datasets
    result = connection.execute(sa.text("SELECT id, details FROM datasets"))

    for row in result:
        dataset_id = row[0]
        details = row[1]

        # Check if 'urn' field exists and is an object
        if 'urn' in details and isinstance(details['urn'], dict):
            try:
                urn_dict = details['urn']

                # Convert back to short URN format
                agency_id = urn_dict.get('agency_id', '')
                resource_id = urn_dict.get('resource_id', '')
                version = urn_dict.get('version', 'latest')

                short_urn = f"{agency_id}:{resource_id}({version})"

                # Update the details with the old URN string format
                details['urn'] = short_urn

                # Update the database - serialize dict to JSON string for JSONB column
                connection.execute(
                    sa.text("UPDATE datasets SET details = CAST(:details AS jsonb) WHERE id = :id"),
                    {"details": json.dumps(details), "id": dataset_id},
                )
            except (KeyError, TypeError) as e:
                # Log or handle conversion errors gracefully
                logger.warning(f"Failed to convert URN for dataset {dataset_id}: {e}")

"""Update dataset dimensions config

Revision ID: 5fac4226f0af
Revises: 9e0af00d4447
Create Date: 2025-12-11 11:16:48.886861

"""

import json
from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '5fac4226f0af'
down_revision: str | None = '9e0af00d4447'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    connection = op.get_bind()

    result = connection.execute(sa.text("SELECT id, details FROM datasets"))
    rows = result.fetchall()

    for dataset_id, old_details in rows:
        new_details = transform_old_to_new(old_details)
        connection.execute(
            sa.text("UPDATE datasets SET details = :details WHERE id = :id"),
            {"details": json.dumps(new_details), "id": dataset_id},
        )


def downgrade() -> None:
    connection = op.get_bind()

    result = connection.execute(sa.text("SELECT id, details FROM datasets"))
    rows = result.fetchall()

    for dataset_id, new_details in rows:
        old_details = transform_new_to_old(new_details)
        connection.execute(
            sa.text("UPDATE datasets SET details = :details WHERE id = :id"),
            {"details": json.dumps(old_details), "id": dataset_id},
        )


def transform_old_to_new(old_details: dict) -> dict:
    """Transform old format to new format according to specification."""

    fields_to_remove = {
        'fixedIndicator',
        'countryDimension',
        'countryDimensionAlias',
        'dimensionAliases',
        'specialDimensions',
        'virtualDimensions',
        'dimensionAllValues',
        'frequencyDimension',
        'indicatorDimensions',
        'dimensionDefaultQueries',
        'indicatorDimensionsRequiredForQuery',
    }

    # Copy all fields except those to be transformed
    new_details = {k: v for k, v in old_details.items() if k not in fields_to_remove}

    new_details['dimensions'] = construct_dimensions(old_details)
    return new_details


def construct_dimensions(old_details: dict) -> dict[Any, Any]:
    dimensions = {}

    # Collect all dimension IDs from various sources
    country_dim = old_details.get('countryDimension')
    frequency_dim = old_details.get('frequencyDimension')
    indicator_dims = set(old_details.get('indicatorDimensions', []))
    required_indicators = set(old_details.get('indicatorDimensionsRequiredForQuery', []))
    all_values_dims = set(old_details.get('dimensionAllValues', {}).keys())
    virtual_dims = {v.get('id'): v for v in old_details.get('virtualDimensions', [])}
    default_queries = old_details.get('dimensionDefaultQueries', {})
    special_dims = {
        sd['dimension_id']: sd['processor_id'] for sd in old_details.get('specialDimensions', [])
    }

    # Collect all unique dimension IDs
    all_dim_ids = set()
    if country_dim:
        all_dim_ids.add(country_dim)
    if frequency_dim:
        all_dim_ids.add(frequency_dim)
    all_dim_ids.update(indicator_dims)
    all_dim_ids.update(all_values_dims)
    all_dim_ids.update(virtual_dims.keys())
    all_dim_ids.update(default_queries.keys())
    all_dim_ids.update(special_dims.keys())

    # Build each dimension entry
    for dim_id in all_dim_ids:
        dim_config: dict[str, Any] = {}

        # Determine dimension type
        if dim_id == country_dim:
            dim_config['dimensionType'] = 'NON_INDICATOR'
            dim_config['subtype'] = 'REGION'
            country_alias = old_details.get('countryDimensionAlias')
            if country_alias:
                dim_config['alias'] = country_alias

        elif dim_id == frequency_dim:
            dim_config['dimensionType'] = 'NON_INDICATOR'
            dim_config['subtype'] = 'FREQUENCY'

        elif dim_id in indicator_dims:
            dim_config['dimensionType'] = 'INDICATOR'
            dim_config['isRequired'] = dim_id in required_indicators

        elif dim_id == 'TIME_PERIOD':
            dim_config['dimensionType'] = 'TIME_PERIOD'

        elif dim_id in special_dims:
            dim_config['dimensionType'] = 'SPECIAL'
            dim_config['processorId'] = special_dims[dim_id]

        else:
            dim_config['dimensionType'] = 'NON_INDICATOR'

        # Add additional properties

        if dim_id in old_details.get('dimensionAllValues', {}):
            dim_config['allValues'] = old_details['dimensionAllValues'][dim_id]

        if dim_id in virtual_dims:
            virtual_data = virtual_dims[dim_id]
            dim_config['virtual'] = virtual_data

        if dim_id in default_queries:
            dim_config['defaultQueries'] = default_queries[dim_id]

        dimensions[dim_id] = dim_config
    return dimensions


def transform_new_to_old(new_details: dict) -> dict:
    """Transform new format back to old format for downgrade."""
    old_details = {k: v for k, v in new_details.items() if k != 'dimensions'}

    # Extract from dimensions
    dimensions = new_details.get('dimensions', {})

    old_details['indicatorDimensions'] = []
    old_details['indicatorDimensionsRequiredForQuery'] = []
    old_details['specialDimensions'] = []
    old_details['virtualDimensions'] = []
    old_details['dimensionAllValues'] = {}
    old_details['dimensionDefaultQueries'] = {}
    old_details['dimensionAliases'] = {}

    for dim_id, dim_config in dimensions.items():
        dim_type = dim_config.get('dimensionType')

        if dim_type == 'NON_INDICATOR':
            subtype = dim_config.get('subtype')
            if subtype == 'REGION':
                old_details['countryDimension'] = dim_id
                if 'alias' in dim_config:
                    old_details['countryDimensionAlias'] = dim_config['alias']
            elif subtype == 'FREQUENCY':
                old_details['frequencyDimension'] = dim_id

        elif dim_type == 'INDICATOR':
            old_details['indicatorDimensions'].append(dim_id)
            if dim_config.get('isRequired'):
                old_details['indicatorDimensionsRequiredForQuery'].append(dim_id)

        elif dim_type == 'SPECIAL':
            old_details['specialDimensions'].append(
                {'dimension_id': dim_id, 'processor_id': dim_config.get('processorId')}
            )

        # Extract additional properties
        if 'allValues' in dim_config:
            old_details['dimensionAllValues'][dim_id] = dim_config['allValues']

        if 'virtual' in dim_config:
            virtual = dim_config['virtual']
            old_details['virtualDimensions'].append(
                {
                    'id': dim_id,
                    'name': virtual.get('name'),
                    'description': virtual.get('description'),
                    'value': virtual.get('value'),
                }
            )

        if 'defaultQueries' in dim_config:
            old_details['dimensionDefaultQueries'][dim_id] = dim_config['defaultQueries']

    return old_details

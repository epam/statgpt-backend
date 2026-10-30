"""store discovery indicators with their units

Revision ID: 4eb4d43ba47d
Revises: 492ab5673baa
Create Date: 2026-10-30 15:24:36.064256

"""

import re
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4eb4d43ba47d'
down_revision: str | None = '492ab5673baa'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# A copy of the flat form's rules as they stand, so this revision keeps converting the same way
# whatever the application's parser later becomes.
_TRAILING_UNIT = re.compile(r"(?P<name>.*)\((?P<unit>[^()]*)\)")
_WHITESPACE = re.compile(r"\s+")
_SEPARATOR = "; "

_TEXT_TABLE = sa.table(
    'discovery_datasets',
    sa.column('id', sa.Integer()),
    sa.column('indicators_coverage', sa.String()),
)
_JSON_TABLE = sa.table(
    'discovery_datasets',
    sa.column('id', sa.Integer()),
    sa.column('indicators_coverage', postgresql.JSONB()),
)


def _normalize(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _parse(cell: str) -> list[dict[str, str]]:
    indicators = []
    for token in cell.split(";"):
        entry = _normalize(token)
        if not entry:
            continue
        name, units = entry, ""
        if match := _TRAILING_UNIT.fullmatch(entry):
            if (head := match["name"].strip()) and (unit := _normalize(match["unit"])):
                name, units = head, unit
        indicators.append({"name": name, "units": units})
    return indicators


def _render(indicators: list[dict[str, str]]) -> str:
    return _SEPARATOR.join(
        f"{item['name']} ({item['units']})" if item['units'] else item['name']
        for item in indicators
    )


def upgrade() -> None:
    # The flat text cannot be converted in SQL, so it is read out first and written back once
    # the column holds JSON. A name that ends in parentheses of its own and was stored without
    # units is read as having them: the flat form cannot tell the two apart, and only a
    # re-upload of the record's indicators sheet does.
    connection = op.get_bind()
    rows = connection.execute(sa.select(_TEXT_TABLE.c.id, _TEXT_TABLE.c.indicators_coverage)).all()

    op.alter_column(
        'discovery_datasets',
        'indicators_coverage',
        existing_type=sa.VARCHAR(),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=False,
        postgresql_using="'[]'::jsonb",
    )

    for row_id, cell in rows:
        if indicators := _parse(cell):
            connection.execute(
                _JSON_TABLE.update()
                .where(_JSON_TABLE.c.id == row_id)
                .values(indicators_coverage=indicators)
            )

    # No verdict is reset: every indexing run validates each record again, so the first one
    # after this revision reports the records whose indicators lack units.


def downgrade() -> None:
    connection = op.get_bind()
    rows = connection.execute(sa.select(_JSON_TABLE.c.id, _JSON_TABLE.c.indicators_coverage)).all()

    op.alter_column(
        'discovery_datasets',
        'indicators_coverage',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=sa.VARCHAR(),
        existing_nullable=False,
        postgresql_using="''",
    )

    for row_id, indicators in rows:
        if indicators:
            connection.execute(
                _TEXT_TABLE.update()
                .where(_TEXT_TABLE.c.id == row_id)
                .values(indicators_coverage=_render(indicators))
            )

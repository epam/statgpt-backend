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

# The flat form's rules live here alone: the application no longer reads that form, so this
# revision is the only place that has to know how it was written.
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


def _split_units(entry: str) -> tuple[str, str]:
    """`(name, units)` of one entry, the units being its last balanced `(...)`.

    Balanced, so units holding parentheses of their own stay whole: `CPI (Index (2015=100))`
    is `CPI` in `Index (2015=100)`. An entry not ending in such a group, or one that is nothing
    but the group, or whose group is empty, has no units.
    """
    if not entry.endswith(")"):
        return entry, ""
    depth = 0
    for position in range(len(entry) - 1, -1, -1):
        if entry[position] == ")":
            depth += 1
        elif entry[position] == "(":
            depth -= 1
            if depth == 0:
                name = entry[:position].strip()
                units = _normalize(entry[position + 1 : -1])
                return (name, units) if name and units else (entry, "")
    # More closing parentheses than opening ones.
    return entry, ""


def _parse(cell: str) -> list[dict[str, str]]:
    indicators = []
    for token in cell.split(";"):
        if entry := _normalize(token):
            name, units = _split_units(entry)
            indicators.append({"name": name, "units": units})
    return indicators


def _render_one(item: dict[str, str]) -> str:
    # A ';' separates the entries of the flat form, so one inside a value becomes ','.
    name = item['name'].replace(";", ",")
    units = item.get('units', "").replace(";", ",")
    return f"{name} ({units})" if units else name


def _render(indicators: list[dict[str, str]]) -> str:
    return _SEPARATOR.join(_render_one(item) for item in indicators)


def upgrade() -> None:
    # The flat text cannot be converted in SQL, so it is read out first and written back once
    # the column holds JSON. A name that ends in parentheses of its own and was stored without
    # units is read as having them: the flat form cannot tell the two apart, and only a
    # re-upload of the record's indicators sheet does. A converted list renders back to the
    # text a workbook upload stored, so the summaries fingerprinted on that text are kept.
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

    params = [
        {"row_id": row_id, "indicators": indicators}
        for row_id, cell in rows
        if (indicators := _parse(cell))
    ]
    if params:
        connection.execute(
            _JSON_TABLE.update()
            .where(_JSON_TABLE.c.id == sa.bindparam("row_id"))
            .values(indicators_coverage=sa.bindparam("indicators", type_=postgresql.JSONB())),
            params,
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

    params = [
        {"row_id": row_id, "indicators": _render(indicators)}
        for row_id, indicators in rows
        if indicators
    ]
    if params:
        connection.execute(
            _TEXT_TABLE.update()
            .where(_TEXT_TABLE.c.id == sa.bindparam("row_id"))
            .values(indicators_coverage=sa.bindparam("indicators")),
            params,
        )

import re


def _replace_illegal_characters(name: str) -> str:
    """Replace illegal characters in Postgres table name."""

    return re.sub("[^a-zA-Z0-9_]+", "_", name)


def to_postgres_table_name(name: str) -> str:
    """Create a valid Postgres table name"""

    name = _replace_illegal_characters(name)
    name = re.sub("_+", "_", name)  # Replace multiple underscores with a single one
    name = name.strip("_")  # Remove leading and trailing underscores
    name = name[:63]  # Apply table name length limit

    return name

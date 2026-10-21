import csv
from collections.abc import Iterator

import pytest

_PYTHON_CSV_FIELD_LIMIT = 131_072


@pytest.fixture
def default_csv_field_limit() -> Iterator[int]:
    """Start from Python's own CSV field limit, and put the process-wide one back after."""
    previous = csv.field_size_limit(_PYTHON_CSV_FIELD_LIMIT)
    yield _PYTHON_CSV_FIELD_LIMIT
    csv.field_size_limit(previous)

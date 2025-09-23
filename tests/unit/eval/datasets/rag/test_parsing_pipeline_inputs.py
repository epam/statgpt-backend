from datetime import date

import pytest

from eval.dataset.rag.target_gathering_agent.schemas import RAGAgentInputs


@pytest.mark.parametrize(
    "user_input_end_date, user_input_current_date, expected_output",
    [
        ('', date(2025, 8, 1), date(2025, 8, 1)),
        (None, date(2025, 8, 1), date(2025, 8, 1)),
        ('2025-07-01', date(2025, 8, 1), date(2025, 7, 1)),
        ('2025-08-02', date(2025, 8, 1), date(2025, 8, 1)),
        ('2025-09-01', date(2025, 8, 1), date(2025, 8, 1)),
        (date(2025, 7, 1), date(2025, 8, 1), date(2025, 7, 1)),
        (date(2025, 8, 2), date(2025, 8, 1), date(2025, 8, 1)),
        (date(2025, 9, 1), date(2025, 8, 1), date(2025, 8, 1)),
    ],
)
def test_ensure_end_date(
    user_input_end_date: str | date | None, user_input_current_date: date, expected_output: date
):
    result = RAGAgentInputs.ensure_end_date(
        orig_end_date=user_input_end_date, current_date=user_input_current_date
    )
    assert result == expected_output

"""Unit tests for the exception utilities."""

from statgpt.common.utils.exceptions import format_exception_reason


class TestFormatExceptionReason:
    """Tests for ``format_exception_reason``."""

    def test_plain_exception(self) -> None:
        assert format_exception_reason(ValueError("bad value")) == "ValueError: bad value"

    def test_empty_message_falls_back_to_class_name(self) -> None:
        assert format_exception_reason(KeyError()) == "KeyError"

    def test_single_child_group_unwraps_to_leaf(self) -> None:
        # The original bug: str() of this group yields the generic
        # "unhandled errors in task group (1 sub-exception)" message.
        group = ExceptionGroup("group", [ValueError("dataset 'X' not found")])
        assert format_exception_reason(group) == "ValueError: dataset 'X' not found"

    def test_multi_child_group_joins_leaves(self) -> None:
        group = ExceptionGroup("group", [ValueError("first"), ConnectionError("timeout")])
        assert format_exception_reason(group) == "ValueError: first; ConnectionError: timeout"

    def test_nested_group_is_flattened(self) -> None:
        group = ExceptionGroup(
            "outer",
            [ValueError("first"), ExceptionGroup("inner", [RuntimeError("second")])],
        )
        assert format_exception_reason(group) == "ValueError: first; RuntimeError: second"

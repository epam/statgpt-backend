import pytest

from statgpt.common.utils.markdown import format_as_markdown_list


class TestFormatAsMarkdownList:
    """Tests for the format_as_markdown_list utility function."""

    # ---- Unordered list ----

    def test_unordered_list(self):
        result = format_as_markdown_list(["apple", "banana", "cherry"])
        assert result == "- apple\n- banana\n- cherry"

    def test_unordered_list_single_item(self):
        result = format_as_markdown_list(["only item"])
        assert result == "- only item"

    def test_unordered_is_default(self):
        """Calling without list_type should default to unordered."""
        result = format_as_markdown_list(["a", "b"])
        assert result == "- a\n- b"

    # ---- Ordered list ----

    def test_ordered_list(self):
        result = format_as_markdown_list(["first", "second", "third"], list_type="ordered")
        assert result == "1. first\n2. second\n3. third"

    def test_ordered_list_single_item(self):
        result = format_as_markdown_list(["only item"], list_type="ordered")
        assert result == "1. only item"

    def test_ordered_list_numbering_starts_at_one(self):
        """Verify that ordered numbering begins at 1, not 0."""
        result = format_as_markdown_list(["a"], list_type="ordered")
        assert result.startswith("1.")

    # ---- Empty list ----

    def test_empty_list_unordered(self):
        result = format_as_markdown_list([])
        assert result == ""

    def test_empty_list_ordered(self):
        result = format_as_markdown_list([], list_type="ordered")
        assert result == ""

    # ---- Invalid list_type ----

    def test_invalid_list_type_raises_value_error(self):
        with pytest.raises(ValueError, match="list_type must be either"):
            format_as_markdown_list(["item"], list_type="bullet")

    # ---- Content preservation ----

    def test_preserves_special_characters(self):
        """Markdown special characters in items should be passed through as-is."""
        items = ["**bold**", "[link](url)", "`code`"]
        result = format_as_markdown_list(items)
        assert "- **bold**" in result
        assert "- [link](url)" in result
        assert "- `code`" in result

    def test_preserves_whitespace_in_items(self):
        """Leading/trailing whitespace within items should be preserved."""
        result = format_as_markdown_list(["  padded  "])
        assert result == "-   padded  "

    def test_multiline_items(self):
        """Items containing newlines should appear on the same bullet line."""
        result = format_as_markdown_list(["line1\nline2"])
        assert result == "- line1\nline2"

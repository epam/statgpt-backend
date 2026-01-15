"""Tests for CLI console utilities."""

import pytest

from statgpt.cli.shared.console import mask_secret


class TestMaskSecret:
    """Tests for mask_secret function."""

    def test_none_value(self):
        """None value should return 'not set' indicator."""
        result = mask_secret(None)
        assert result == "[dim]\u2717 not set[/dim]"

    def test_short_value_fully_masked(self):
        """Value shorter than or equal to visible_chars should be fully masked."""
        result = mask_secret("abc")
        assert result == "\u2022\u2022\u2022"  # 3 bullets

    def test_exact_boundary_fully_masked(self):
        """Value exactly at visible_chars length should be fully masked."""
        result = mask_secret("1234")  # Default visible_chars=4
        assert result == "\u2022\u2022\u2022\u2022"  # 4 bullets

    def test_normal_value_partial_mask(self):
        """Normal value should show only last few characters."""
        result = mask_secret("secretvalue")
        # 12 bullets + last 4 chars
        assert result == "\u2022" * 12 + "alue"

    def test_custom_visible_chars(self):
        """Custom visible_chars should control visible portion."""
        result = mask_secret("mysecret", visible_chars=2)
        # 12 bullets + last 2 chars
        assert result == "\u2022" * 12 + "et"

    def test_visible_chars_larger_than_value(self):
        """When visible_chars >= len(value), fully mask."""
        result = mask_secret("abc", visible_chars=5)
        assert result == "\u2022\u2022\u2022"

    def test_one_char_visible(self):
        """Single visible character at end."""
        result = mask_secret("secret", visible_chars=1)
        assert result == "\u2022" * 12 + "t"

    def test_empty_string(self):
        """Empty string should be fully masked (0 bullets)."""
        result = mask_secret("")
        # len("") = 0 <= 4, so fully masked with 0 bullets
        assert result == ""

    def test_long_value(self):
        """Long value should still show only last visible_chars."""
        long_value = "a" * 100 + "end1"
        result = mask_secret(long_value)
        assert result == "\u2022" * 12 + "end1"


class TestMaskSecretEdgeCases:
    """Edge case tests for mask_secret function."""

    @pytest.mark.parametrize(
        "value,visible,expected_visible",
        [
            ("password123", 4, "d123"),
            ("api_key_12345", 5, "12345"),
            ("x", 4, None),  # Fully masked
            ("ab", 4, None),  # Fully masked
            ("abcd", 4, None),  # Fully masked at boundary
            ("abcde", 4, "bcde"),  # One char over, shows last 4
        ],
    )
    def test_boundary_cases(self, value, visible, expected_visible):
        """Test boundary cases for masking."""
        result = mask_secret(value, visible_chars=visible)
        if expected_visible is None:
            # Should be fully masked
            assert result == "\u2022" * len(value)
        else:
            assert result.endswith(expected_visible)
            assert result.startswith("\u2022" * 12)

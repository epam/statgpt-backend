import base64

import pytest

from statgpt.common.utils.misc import (
    argparse_parse_int_or_none,
    batched,
    create_base64_uuid,
    crc32_hash,
    crc32_hash_incremental,
    str2bool,
)


# ---------------------------------------------------------------------------
# batched
# ---------------------------------------------------------------------------


class TestBatched:
    def test_exact_batches(self):
        result = list(batched("ABCDEF", 3))
        assert result == [["A", "B", "C"], ["D", "E", "F"]]

    def test_last_batch_shorter(self):
        result = list(batched("ABCDEFG", 3))
        assert result == [["A", "B", "C"], ["D", "E", "F"], ["G"]]

    def test_batch_size_one(self):
        result = list(batched([1, 2, 3], 1))
        assert result == [[1], [2], [3]]

    def test_batch_size_larger_than_input(self):
        result = list(batched([1, 2], 10))
        assert result == [[1, 2]]

    def test_empty_iterable(self):
        result = list(batched([], 3))
        assert result == []

    def test_works_with_generator(self):
        gen = (x for x in range(5))
        result = list(batched(gen, 2))
        assert result == [[0, 1], [2, 3], [4]]


# ---------------------------------------------------------------------------
# crc32_hash
# ---------------------------------------------------------------------------


class TestCrc32Hash:
    def test_deterministic(self):
        """Same input should always produce the same hash."""
        assert crc32_hash("hello") == crc32_hash("hello")

    def test_different_inputs_different_hashes(self):
        assert crc32_hash("hello") != crc32_hash("world")

    def test_returns_positive_integer(self):
        result = crc32_hash("test")
        assert isinstance(result, int)
        assert result >= 0

    def test_empty_string(self):
        result = crc32_hash("")
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# crc32_hash_incremental
# ---------------------------------------------------------------------------


class TestCrc32HashIncremental:
    def test_deterministic(self):
        values = ["alpha", "beta", "gamma"]
        assert crc32_hash_incremental(values) == crc32_hash_incremental(values)

    def test_different_values_different_hashes(self):
        assert crc32_hash_incremental(["a", "b"]) != crc32_hash_incremental(["c", "d"])

    def test_order_matters(self):
        """Changing the order of values should produce a different hash."""
        assert crc32_hash_incremental(["a", "b"]) != crc32_hash_incremental(["b", "a"])

    def test_empty_iterable(self):
        """Empty input should return 0 (initial CRC value masked to unsigned)."""
        result = crc32_hash_incremental([])
        assert result == 0

    def test_returns_positive_integer(self):
        result = crc32_hash_incremental(["value"])
        assert isinstance(result, int)
        assert result >= 0


# ---------------------------------------------------------------------------
# str2bool
# ---------------------------------------------------------------------------


class TestStr2Bool:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("  true  ", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("  false  ", False),
            ("", False),
            ("yes", False),
            ("1", False),
            ("0", False),
        ],
    )
    def test_str2bool(self, value, expected):
        assert str2bool(value) == expected


# ---------------------------------------------------------------------------
# argparse_parse_int_or_none
# ---------------------------------------------------------------------------


class TestArgparseParseIntOrNone:
    def test_empty_string_returns_none(self):
        assert argparse_parse_int_or_none("") is None

    def test_valid_positive_int(self):
        assert argparse_parse_int_or_none("42") == 42

    def test_valid_zero(self):
        assert argparse_parse_int_or_none("0") == 0

    def test_valid_negative_int(self):
        assert argparse_parse_int_or_none("-7") == -7

    def test_large_number(self):
        assert argparse_parse_int_or_none("999999999") == 999999999

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            argparse_parse_int_or_none("abc")

    def test_float_string_raises(self):
        with pytest.raises(ValueError):
            argparse_parse_int_or_none("3.14")


# ---------------------------------------------------------------------------
# create_base64_uuid
# ---------------------------------------------------------------------------


class TestCreateBase64Uuid:
    def test_returns_string(self):
        result = create_base64_uuid()
        assert isinstance(result, str)

    def test_length_is_22(self):
        """128-bit UUID in URL-safe base64 without padding is 22 chars."""
        result = create_base64_uuid()
        assert len(result) == 22

    def test_no_padding_characters(self):
        result = create_base64_uuid()
        assert "=" not in result

    def test_url_safe_characters(self):
        """Result should only contain URL-safe base64 characters."""
        result = create_base64_uuid()
        assert "+" not in result
        assert "/" not in result

    def test_unique_values(self):
        """Two calls should produce different UUIDs."""
        a = create_base64_uuid()
        b = create_base64_uuid()
        assert a != b

    def test_decodable(self):
        """Result should be decodable back to 16 bytes (128-bit UUID)."""
        result = create_base64_uuid()
        # Add back padding for decoding
        padded = result + "=="
        decoded = base64.urlsafe_b64decode(padded)
        assert len(decoded) == 16

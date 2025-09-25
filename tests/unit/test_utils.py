import os
import time
from datetime import date

import pytest

from common import utils as common_utils
from common.config import utils as common_config_utils
from statgpt import utils as statgpt_utils


@pytest.mark.parametrize(
    "secret, target",
    (
        (None, None),
        ("ab", "**"),
        ("1234", "****"),
        ("12345", "1***5"),
        ("123456", "1***6"),
        ("1234567", "12***67"),
        ("12345678", "12***78"),
        ("123456789", "123***789"),
        ("1234567890", "123***890"),
    ),
)
def test_secret_2_safe_str(secret, target):
    out = common_utils.secret_2_safe_str(secret)
    assert out == target


@pytest.mark.parametrize(
    "date, target",
    (
        (date(2024, 1, 1), "2024"),
        (date(2023, 7, 25), "2023"),
    ),
)
def test_format_date_freq_a(date, target):
    out = statgpt_utils.format_date_freq_a(date)
    assert out == target


@pytest.mark.parametrize(
    "date, target",
    (
        (date(2024, 1, 1), "2024-Q1"),
        (date(2024, 2, 1), "2024-Q1"),
        (date(2024, 3, 1), "2024-Q1"),
        (date(2024, 4, 1), "2024-Q2"),
        (date(2024, 5, 1), "2024-Q2"),
        (date(2024, 6, 1), "2024-Q2"),
        (date(2023, 7, 25), "2023-Q3"),
        (date(2023, 8, 25), "2023-Q3"),
        (date(2023, 9, 25), "2023-Q3"),
        (date(2023, 10, 25), "2023-Q4"),
        (date(2023, 11, 25), "2023-Q4"),
        (date(2023, 12, 25), "2023-Q4"),
    ),
)
def test_format_date_freq_q(date, target):
    out = statgpt_utils.format_date_freq_q(date)
    assert out == target


@pytest.mark.parametrize(
    "date, target",
    (
        (date(2024, 1, 1), "2024-M01"),
        (date(2024, 2, 1), "2024-M02"),
        (date(2024, 3, 1), "2024-M03"),
        (date(2024, 4, 1), "2024-M04"),
        (date(2024, 5, 1), "2024-M05"),
        (date(2024, 6, 1), "2024-M06"),
        (date(2023, 7, 25), "2023-M07"),
        (date(2023, 8, 25), "2023-M08"),
        (date(2023, 9, 25), "2023-M09"),
        (date(2023, 10, 25), "2023-M10"),
        (date(2023, 11, 25), "2023-M11"),
        (date(2023, 12, 25), "2023-M12"),
    ),
)
def test_format_date_freq_m(date, target):
    out = statgpt_utils.format_date_freq_m(date)
    assert out == target


@pytest.mark.parametrize(
    "s, sep, target",
    (
        ("", "/", [""]),
        ("some text", "/", ["some text"]),
        ("a", "/", ["a"]),
        ("a/b", "/", ["a", "a/b"]),
        ("a/b/c", "/", ["a", "a/b", "a/b/c"]),
    ),
)
def test_string_split_snowball(s, sep, target):
    res = common_utils.string_split_snowball(s, sep)
    res = list(res)
    assert res == target


@pytest.mark.parametrize(
    "ref_date, quarter_offset, target",
    (
        (date(2024, 1, 1), -5, "2022-Q4"),
        (date(2024, 1, 1), -4, "2023-Q1"),
        (date(2024, 1, 1), -3, "2023-Q2"),
        (date(2024, 1, 1), -2, "2023-Q3"),
        (date(2024, 1, 1), -1, "2023-Q4"),
        (date(2024, 1, 1), 0, "2024-Q1"),
        (date(2024, 1, 1), 1, "2024-Q2"),
        (date(2024, 1, 1), 2, "2024-Q3"),
        (date(2024, 7, 1), -1, "2024-Q2"),
        (date(2024, 7, 1), 0, "2024-Q3"),
        (date(2024, 7, 1), 1, "2024-Q4"),
    ),
)
def test_get_relative_quarter(ref_date, quarter_offset, target):
    res = statgpt_utils.get_relative_quarter(ref_date=ref_date, quarter_offset=quarter_offset)
    assert res == target


@pytest.mark.parametrize(
    "value, target",
    (
        ("", ""),
        ("test", "test"),
        ("VAR1", "VAR1"),
        # No replacement needed since env variables are not properly formatted
        ("${VAR1}", "${VAR1}"),
        ("${VAR1|default_1}", "${VAR1|default_1}"),
        # Correctly formatted environment variables
        ("$env:{VAR_NOT_EXIST}", "$env:{VAR_NOT_EXIST}"),
        ("$env:{VAR1}", "value1"),
        ("$env:{VAR1} $env:{VAR2}", "value1 value2"),
        (
            "key1:$env:{VAR1}\nkey2:$env:{VAR2}\nkey3:$env:{VAR3}",
            "key1:value1\nkey2:value2\nkey3:value3",
        ),
        ("$env:{VAR1} $env:{VAR2} $env:{VAR3} $env:{VAR4}", "value1 value2 value3 value4"),
        ({"key": "$env:{VAR1}"}, {"key": "value1"}),
        ({"key": {"key": "$env:{VAR1}"}}, {"key": {"key": "value1"}}),
        ({"key": "$env:{VAR1}"}, {"key": "value1"}),
        ({"key": {"key": "$env:{VAR1}"}}, {"key": {"key": "value1"}}),
        ({"key": "$env:{VAR1}"}, {"key": "value1"}),
        ({"key": {"key": "$env:{VAR1}"}}, {"key": {"key": "value1"}}),
        ({"key": "$env:{VAR1}"}, {"key": "value1"}),
        ({"key": {"key": "$env:{VAR1}"}}, {"key": {"key": "value1"}}),
        # yaml multi-level text content
        (
            "key1: $env:{VAR1}\nkey2: $env:{VAR2}\nkey3: $env:{VAR3}",
            "key1: value1\nkey2: value2\nkey3: value3",
        ),
        # env vars with defaults
        ("$env:{VAR1|default_value}", "value1"),
        ("$env:{VAR_NOT_EXIST|default_value}", "default_value"),
        (
            "$env:{VAR_NOT_EXIST|default_1} $env:{VAR2} $env:{VAR3|default3}",
            "default_1 value2 value3",
        ),
        ("$env:{VAR1}|$env:{VAR2}", "value1|value2"),
    ),
)
def test_replace_envs(value, target):
    os.environ["VAR1"] = "value1"
    os.environ["VAR2"] = "value2"
    os.environ["VAR3"] = "value3"
    os.environ["VAR4"] = "value4"

    res = common_config_utils.replace_envs(value)
    assert res == target
    os.environ.pop("VAR1")
    os.environ.pop("VAR2")
    os.environ.pop("VAR3")
    os.environ.pop("VAR4")


@pytest.mark.parametrize(
    "value, prefix, target",
    (
        # Test with special characters
        ("$env:{SPECIAL@VAR}", "SOME_PREFIX_", "special!value"),
        ("$env:{VAR_WITH_SPACES}", "SOME_PREFIX_", "value with spaces"),
        # Test with empty values
        ("$env:{EMPTY_VAR}", "SOME_PREFIX_", ""),
        ("prefix_$env:{EMPTY_VAR}_suffix", "SOME_PREFIX_", "prefix__suffix"),
        # Test with multiple occurrences of same variable
        ("$env:{VAR1} and $env:{VAR1}", "SOME_PREFIX_", "value1 and value1"),
        # Test with JSON-like structure
        (
            '{"key1": "$env:{VAR1}", "key2": "$env:{VAR2}"}',
            "SOME_PREFIX_",
            '{"key1": "value1", "key2": "value2"}',
        ),
        # Test with URL-like structure
        (
            "https://$env:{VAR1}:$env:{VAR2}@example.com",
            "SOME_PREFIX_",
            "https://value1:value2@example.com",
        ),
        # Test with mixed case variables
        ("$env:{VAR_UPPER} $env:{var_lower}", "SOME_PREFIX_", "UPPER_VALUE lower_value"),
        # Test with numeric values
        ("$env:{NUMERIC_VAR}", "SOME_PREFIX_", "12345"),
        # Test with escaped characters
        ("$env:{ESCAPED_VAR}", "SOME_PREFIX_", "value\\with\\backslashes"),
        # Test with multiple prefixes
        ("$env:{OTHER_PREFIX_VAR}", "SOME_PREFIX_", "$env:{OTHER_PREFIX_VAR}"),
        # Test with environment variables in the middle of words
        ("pre$env:{VAR1}post", "SOME_PREFIX_", "prevalue1post"),
        # Test with multiple variables in complex structure
        (
            """
                    server:
                      host: $env:{VAR1}
                      port: $env:{VAR2}
                      credentials:
                        username: $env:{VAR3}
                        password: $env:{VAR4}
                    """,
            "SOME_PREFIX_",
            """
                    server:
                      host: value1
                      port: value2
                      credentials:
                        username: value3
                        password: value4
                    """,
        ),
    ),
)
def test_replace_envs_with_prefix(value, prefix, target):
    # Setup environment variables
    test_vars = {
        "SOME_PREFIX_VAR1": "value1",
        "SOME_PREFIX_VAR2": "value2",
        "SOME_PREFIX_VAR3": "value3",
        "SOME_PREFIX_VAR4": "value4",
        "SOME_PREFIX_SPECIAL@VAR": "special!value",
        "SOME_PREFIX_VAR_WITH_SPACES": "value with spaces",
        "SOME_PREFIX_EMPTY_VAR": "",
        "SOME_PREFIX_VAR_UPPER": "UPPER_VALUE",
        "SOME_PREFIX_var_lower": "lower_value",
        "SOME_PREFIX_NUMERIC_VAR": "12345",
        "SOME_PREFIX_ESCAPED_VAR": "value\\with\\backslashes",
    }

    try:
        # Set environment variables
        for key, val in test_vars.items():
            os.environ[key] = val

        res = common_config_utils.replace_envs(value, prefix)
        assert res == target

    finally:
        # Cleanup environment variables
        for key in test_vars:
            os.environ.pop(key, None)


@pytest.mark.parametrize(
    "url, new_url_prefix, result",
    (
        ("", "https://example.com", ""),
        ("non-dial-url/folder/file.pdf", "https://example.com", "non-dial-url/folder/file.pdf"),
        ("files/folder/file.pdf", "https://example.com", "https://example.com/file.pdf"),
        (
            "files/folder2/file2.pdf",
            "https://example2.com/files",
            "https://example2.com/files/file2.pdf",
        ),
        (
            "files/5ZAg19v7pWkveQsdLvCJ2x/appdata/dial-rag-pgvector/folder_name/file_name.pdf#page=12",
            "https://my-portal.com/publications",
            "https://my-portal.com/publications/file_name.pdf#page=12",
        ),
        ("prefix/files/folder/file.pdf", "https://example.com", "prefix/files/folder/file.pdf"),
        (
            "https://some.url/files/folder2/file2.pdf",
            "https://example2.com/files",
            "https://some.url/files/folder2/file2.pdf",
        ),
    ),
)
def test_replace_dial_url(url, new_url_prefix, result):
    res = statgpt_utils.replace_dial_url(url, new_url_prefix)
    assert res == result


def test_cache():
    cache = common_utils.Cache(ttl=2)  # 2 seconds TTL

    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    cache.set("key2", "value2")
    assert cache.get("key2") == "value2"

    assert cache.get("non_existent_key") is None

    time.sleep(3)  # Wait for items to expire

    cache.set("key3", "value3")

    assert cache.get("key1") is None
    assert cache.get("key2") is None
    assert cache.get("key3") == "value3"

import os
import re
from typing import TypeVar

from pydantic import SecretStr


def _replace_envs_dict(d: dict, prefix: str | None = None) -> dict:
    for key, value in d.items():
        if isinstance(value, str):
            d[key] = replace_env(value, prefix)
        elif isinstance(value, dict):
            d[key] = _replace_envs_dict(value, prefix)
    return d


def _replace_envs_str(value: str, prefix: str | None = None) -> str:
    """
    Finds all substrings that match pattern `$env:{env_name}` or `$env:{env_name|default_value}`
    and replaces them with the value of the environment variable.

    If the environment variable is not found, but the default value is provided,
    the substring will be replaced with the default value.

    :param value: String to replace envs in
    :return: string with replaced envs
    """

    pattern = r"\$env:{([^}|]*)(\|([^}|]*))?}"
    matches = re.finditer(pattern, value)

    result = value
    for match in matches:
        full_match = match.group(0)  # The entire match including $env:{}
        env_name = match.group(1)  # Just the variable name
        default_value = match.group(3)  # The default value if provided, otherwise None
        full_env_name = f"{prefix}{env_name}" if prefix else env_name

        # Get environment variable value or default value if not found
        env_value_or_default = os.environ.get(full_env_name, default_value)
        if env_value_or_default is None:
            continue

        # Replace the pattern with the environment variable value
        result = result.replace(full_match, env_value_or_default)

    return result


def replace_env(value: str, prefix: str | None = None) -> str:
    """
    Replaces environment variables in a string.
    Finds all substrings that match pattern `$env:{env_name}` or `$env:{env_name|default_value}`
    and replaces them with the value of the environment variable.

    If the environment variable is not found, but the default value is provided,
    the substring will be replaced with the default value.

    :param value: string to replace envs in
    :param prefix: in case prefix is defined only environment variables with this prefix will be replaced; prefix will
    be removed from the variable name
    :return: string with replaced envs
    """
    return _replace_envs_str(value, prefix)


T = TypeVar('T', str, dict)


def replace_envs(value: T, prefix: str | None = None) -> T:
    """
    Replaces environment variables in either a string or a dictionary.
    For strings: replaces patterns like $env:{env_name} with their environment variable values
    For dictionaries: recursively processes all string values in the dictionary

    :param value: Either a string containing environment variables or a dictionary with string or dictionary values
    :param prefix: in case prefix is defined only environment variables with this prefix will be replaced; prefix will
    be removed from the variable name
    :return: Value with all environment variables replaced
    :raises ValueError: If an environment variable is not found
    """
    if isinstance(value, dict):
        return _replace_envs_dict(value, prefix)
    elif isinstance(value, str):
        return _replace_envs_str(value, prefix)
    else:
        raise ValueError(f"Expected a string or a dictionary, got {type(value)}")


def get_int_env(name: str, default: int | None = None) -> int | None:
    val = os.getenv(name)
    if val is None:
        return default
    return int(val)


def get_secret_env(name: str) -> SecretStr | None:
    val = os.getenv(name)
    if val is None:
        return None
    return SecretStr(val)


def get_bool_env(name: str, default: bool | None = None) -> bool | None:
    val = os.getenv(name)
    if val is None:
        return default

    val = val.strip().lower()
    if val in ["true", '1', 'yes', 'y', 'on']:
        return True

    if val in ["false", '0', 'no', 'n', 'off']:
        return False

    raise ValueError(f"Invalid boolean value for environment variable {name}: {val}")

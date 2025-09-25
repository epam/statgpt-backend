import re

from sdmx.model.v21 import DataStructureDefinition


def replace_consecutive_dots(s: str) -> str:
    # Match any sequence of two or more dots
    pattern = re.compile(r"\.{2,}")

    def repl(m: re.Match) -> str:
        run_length = len(m.group(0))
        # For N dots, produce (".*" repeated (N-1) times) + "."
        return ".*" * (run_length - 1) + "."

    query_string = pattern.sub(repl, s)

    # Handle leading and trailing dots, but avoid double asterisks
    if query_string.startswith('.'):
        query_string = '*' + query_string
    if query_string.endswith('.'):
        query_string = query_string + '*'

    return query_string


def convert_keys_to_str(dsd: DataStructureDefinition, key_dict: dict[str, list[str]]) -> str:
    """Converts a dict of keys to a query string

    Example:
    {"CURRENCY": ["USD", "JPY"]} -> ".USD+JPY..."
    """

    # Make a ContentConstraint from the key dict
    cc = dsd.make_constraint(key_dict)

    query_string = cc.to_query_string(dsd)

    # Query string may return something like "..57200.." which is not supported by most browsers.
    # Replace all consecutive dots (two or more) with the appropriate number of ".*."
    query_string = replace_consecutive_dots(query_string)

    return query_string

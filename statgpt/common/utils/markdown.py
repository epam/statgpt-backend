import typing as t


def format_as_markdown_list(
    value: list[str], list_type: t.Literal["unordered", "ordered"] = "unordered"
) -> str:
    """
    Format a list of strings as a markdown list.
    :param value: value to format
    :param list_type: type of list to format as (unordered or ordered)
    :return: formatted markdown list
    """
    if list_type == "unordered":
        return "\n".join([f"- {item}" for item in value])
    elif list_type == "ordered":
        return "\n".join([f"{i}. {item}" for i, item in enumerate(value, start=1)])
    else:
        raise ValueError("list_type must be either 'unordered' or 'ordered'")

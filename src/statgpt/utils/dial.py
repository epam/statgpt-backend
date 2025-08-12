import re

MD_CODE_TEMPLATE = """```{markdown_format}
{text}
```
"""


def code_2_markdown(markdown_format, text: str) -> str:
    return MD_CODE_TEMPLATE.format(markdown_format=markdown_format, text=text)


def get_json_markdown(data: str) -> str:
    return code_2_markdown("json", data)


def get_python_code_markdown(data: str) -> str:
    return code_2_markdown("python", data)


def replace_dial_url(url: str, new_url_prefix: str) -> str:
    """Replace a DIAL attachment file url with a new prefix."""
    return re.sub(r'^files/.+/([^/]+)$', fr'{new_url_prefix}/\1', url)

#!/usr/bin/env python3
"""
Print the document-metadata JSON-schema a Generic RAG application must be configured with
to hold discovery dataset records, or the reference-area vocabulary they are searched by.

The schema is derived from `DiscoveryDocumentMetadata`, which is the contract: the
application enforces its own copy, configured in DIAL rather than pushed from here, so this
exists to keep that copy honest instead of hand-maintained.

Paste the output into the application's `applicationProperties.metadata_schema` - in
`dial/core/config/config.json` for local development, in the helm chart for a deployment.
An indexing run refuses to publish into an application whose schema is missing any of the
filterable fields, so a mismatch fails loudly rather than producing documents that search
cannot narrow down.

Usage:
    python scripts/print_discovery_metadata_schema.py
    python scripts/print_discovery_metadata_schema.py --patch dial/core/config/config.json \
        --application statgpt-generic-rag-grade-b-and-c
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from statgpt.common.schemas import (
    ChannelDocumentMetadata,
    DiscoveryDocumentMetadata,
    ReferenceAreaDocumentMetadata,
)

_SCHEMAS: dict[str, tuple[type[ChannelDocumentMetadata], str]] = {
    "discovery": (DiscoveryDocumentMetadata, "statgpt-generic-rag-grade-b-and-c"),
    "reference-areas": (ReferenceAreaDocumentMetadata, "statgpt-generic-rag-reference-areas"),
}
"""Each channel's metadata model and the application it is configured on by default.

Two applications, because the two hold different documents: the records, and the vocabulary a
query's reference areas are resolved against before the records are searched.
"""


_DECODER = json.JSONDecoder()
_WHITESPACE = re.compile(r"[ \t\n\r]*")
_DEFAULT_INDENT = 2


def _skip_whitespace(text: str, position: int) -> int:
    match = _WHITESPACE.match(text, position)
    assert match is not None  # the pattern matches the empty string
    return match.end()


def _member_span(text: str, start: int, key: str) -> tuple[int, int] | None:
    """Locate the value of `key` among the members of the object opening at `start`.

    Walks the object's direct members only, so a same-named key nested deeper is never taken
    for it. The last occurrence wins, as it does when the file is loaded.
    """
    span = None
    position = _skip_whitespace(text, start + 1)
    while text[position] != "}":
        name, position = _DECODER.raw_decode(text, position)
        value_start = _skip_whitespace(text, _skip_whitespace(text, position) + 1)
        _, value_end = _DECODER.raw_decode(text, value_start)
        if name == key:
            span = value_start, value_end
        position = _skip_whitespace(text, value_end)
        if text[position] == ",":
            position = _skip_whitespace(text, position + 1)
    return span


def _render_in_place(text: str, start: int, end: int, value: Any) -> str:
    """Render `value` to replace `text[start:end]`, laid out the way the file already is."""
    newline = "\r\n" if "\r\n" in text else "\n"
    line_start = text.rfind("\n", 0, start) + 1
    indent = text[line_start : _skip_whitespace(text, line_start)]
    old = text[start:end]
    step = _DEFAULT_INDENT
    if "\n" in old:
        inner = old.split("\n", 1)[1]
        step = len(inner) - len(inner.lstrip(" ")) - len(indent) or _DEFAULT_INDENT
    return json.dumps(value, indent=step, ensure_ascii=False).replace("\n", newline + indent)


def _patch(config_path: Path, application: str, schema: dict[str, Any]) -> None:
    """Replace the application's metadata schema, leaving every other byte of the file as is.

    The config is hand-maintained, so rewriting it whole would reformat what this has no
    business touching.
    """
    text = config_path.read_bytes().decode("utf-8")
    applications = json.loads(text).get("applications", {})
    if application not in applications:
        known = ", ".join(sorted(applications)) or "none"
        sys.exit(f"No application {application!r} in {config_path}. Found: {known}.")

    start, end = 0, 0
    object_start = _skip_whitespace(text, 0)
    for key in ("applications", application, "applicationProperties", "metadata_schema"):
        span = _member_span(text, object_start, key)
        if span is None:
            sys.exit(
                f"{application!r} in {config_path} has no applicationProperties.metadata_schema"
                ' to replace. Add an empty one ("metadata_schema": {}) and run this again.'
            )
        start, end = span
        object_start = start

    rendered = _render_in_place(text, start, end, schema)
    if text[start:end] == rendered:
        print(f"The metadata schema of {application!r} in {config_path} is up to date.")
        return
    config_path.write_bytes((text[:start] + rendered + text[end:]).encode("utf-8"))
    print(f"Updated the metadata schema of {application!r} in {config_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patch",
        type=Path,
        help="A DIAL core config.json to write the schema into, instead of printing it.",
    )
    parser.add_argument(
        "--schema",
        choices=sorted(_SCHEMAS),
        default="discovery",
        help="Which channel's schema to render: the discovery records or their reference areas.",
    )
    parser.add_argument(
        "--application",
        help="Which application in that config to patch. Defaults to the schema's own.",
    )
    args = parser.parse_args()

    model, default_application = _SCHEMAS[args.schema]
    schema = model.channel_json_schema()
    if args.patch:
        _patch(args.patch, args.application or default_application, schema)
    else:
        print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()

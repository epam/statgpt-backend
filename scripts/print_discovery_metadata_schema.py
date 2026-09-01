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


def _patch(config_path: Path, application: str, schema: dict[str, Any]) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    applications = config.get("applications", {})
    if application not in applications:
        known = ", ".join(sorted(applications)) or "none"
        sys.exit(f"No application {application!r} in {config_path}. Found: {known}.")

    applications[application].setdefault("applicationProperties", {})
    applications[application]["applicationProperties"]["metadata_schema"] = schema
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
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

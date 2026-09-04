from typing import Any

from fastmcp.tools.base import resolve_serialize_by_alias
from fastmcp.utilities.json_schema import compress_schema
from pydantic import BaseModel


def model_to_output_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build a tool's MCP output schema (JSON Schema) from its structured-content model.

    Mirrors how FastMCP derives a tool's output schema from a return-type annotation, so the
    schemas advertised by the dynamically-built channel tools match the ones FastMCP generates
    for the statically-declared admin tools:

    - ``mode="serialization"`` and the model's ``serialize_by_alias`` config are honored so the
      property names match the serialized ``structuredContent`` the tool returns at runtime
      (which FastMCP serializes with the same ``by_alias``). A camelCase-aliased model therefore
      yields a camelCase schema.
    - Titles are pruned to keep the published contract compact.

    A Pydantic model always serializes to a JSON object, so the result is an object schema and
    no result-wrapping (``x-fastmcp-wrap-result``) is needed.
    """
    by_alias = resolve_serialize_by_alias(model)
    schema = model.model_json_schema(by_alias=by_alias, mode="serialization")
    return compress_schema(schema, prune_titles=True)

"""An MCP tool call cites with ids of its own (#730).

No state comes back from an MCP client, so a call cannot continue the citation ids of the calls
before it. Were every call to start at `citation001`, a client calling a tool twice would read one
id naming two sources.
"""

import re
from types import SimpleNamespace

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.mcp.provider import _build_mcp_inputs


def test_each_mcp_call_cites_with_scoped_ids():
    inputs = _build_mcp_inputs(SimpleNamespace(), SimpleNamespace())  # type: ignore[arg-type]

    citation_id = ChainParameters.get_citation_id_space(inputs).next_id()

    assert re.fullmatch(r"citation001-[a-z0-9]{4}", citation_id)

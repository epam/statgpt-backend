from typing import Annotated

from fastmcp.dependencies import Depends
from fastmcp.exceptions import ToolError
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.app.services.chat_facade import ChannelServiceFacade
from statgpt.common.models.database import get_session_context_manager
from statgpt.common.services import GlossaryOfTermsService
from statgpt.mcp_lite.deps import get_channel_facade
from statgpt.mcp_lite.schemas import GlossaryTermFull, GlossaryTermPreview, GlossaryTerms

from ._provider import mcp_tools


@mcp_tools.tool
async def list_glossary_terms(
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
) -> GlossaryTerms:
    """List glossary terms defined for this channel.

    Use this to discover client-specific vocabulary before composing queries
    or interpreting indicator names. Call `get_glossary_term` for a full
    definition of any term.
    """
    terms = await facade.get_available_terms()
    return GlossaryTerms(terms=[GlossaryTermPreview(term=t.term, domain=t.domain) for t in terms])


@mcp_tools.tool
async def get_glossary_term(
    term: Annotated[str, "Exact term name as returned by `list_glossary_terms`."],
    facade: ChannelServiceFacade = Depends(get_channel_facade),  # type: ignore[arg-type]
    session: AsyncSession = Depends(get_session_context_manager),  # type: ignore[arg-type]
) -> GlossaryTermFull:
    """Retrieve the full definition of a single glossary term.

    Match is case-insensitive on the term name within this channel's glossary.
    """
    service = GlossaryOfTermsService(session, session_lock=None)
    terms = await service.get_term_schemas_by_channel(
        channel_id=facade.channel.id, limit=None, offset=0
    )
    needle = term.strip().casefold()
    for t in terms:
        if t.term.casefold() == needle:
            return GlossaryTermFull(
                term=t.term, definition=t.definition, domain=t.domain, source=t.source
            )
    raise ToolError(f"Glossary term not found in this channel: {term!r}")

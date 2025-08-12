from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

import common.models as models
import common.schemas as schemas
from admin_portal.auth.user import User, require_jwt_auth
from admin_portal.services import AdminPortalGlossaryOfTermsService as GlossaryOfTermsService

terms_router = APIRouter(prefix="/terms", tags=["glossary_of_terms"])
channel_terms_router = APIRouter(prefix="/{channel_id}/terms", tags=["glossary_of_terms"])


@channel_terms_router.get("")
async def get_channel_glossary_terms(
    channel_id: int,
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.ListResponse[schemas.GlossaryTerm]:
    """Returns a list of glossary terms for the channel."""

    service = GlossaryOfTermsService(session)

    terms = await service.get_term_schemas_by_channel(channel_id, limit, offset)
    terms_count = await service.get_terms_count(channel_id)

    return schemas.ListResponse[schemas.GlossaryTerm](
        data=terms,
        limit=limit,
        offset=offset,
        count=len(terms),
        total=terms_count,
    )


@channel_terms_router.post("")
async def add_glossary_term_to_channel(
    channel_id: int,
    data: schemas.GlossaryTermBase,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.GlossaryTerm:
    """Add a new term to the channel glossary."""

    return await GlossaryOfTermsService(session).add_term(channel_id, data)


@channel_terms_router.post("/bulk")
async def add_terms_bulk(
    channel_id: int,
    data: list[schemas.GlossaryTermBase],
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> list[schemas.GlossaryTerm]:
    """Add multiple terms to the channel glossary."""

    return await GlossaryOfTermsService(session).add_terms_bulk(channel_id, data)


@channel_terms_router.delete("/bulk")
async def clear_channel_terms(
    channel_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> list[schemas.GlossaryTerm]:
    """Delete all glossary terms for the channel and return the deleted terms."""

    return await GlossaryOfTermsService(session).delete_terms_bulk(channel_id=channel_id)


@terms_router.post("/bulk")
async def update_terms_bulk(
    data: list[schemas.GlossaryTermUpdateBulk],
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> list[schemas.GlossaryTerm]:
    """Update multiple glossary terms."""

    return await GlossaryOfTermsService(session).update_terms_bulk(data)


@terms_router.delete("/bulk")
async def delete_terms_bulk(
    term_ids: list[int],
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> list[schemas.GlossaryTerm]:
    """Delete multiple glossary terms by their ids and return the deleted terms."""

    return await GlossaryOfTermsService(session).delete_terms_bulk(term_ids=term_ids)


@terms_router.get("/{item_id}")
async def get_glossary_term_by_id(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.GlossaryTerm:
    """Returns a glossary term by id."""

    return await GlossaryOfTermsService(session).get_term_schema_by_id(item_id)


@terms_router.post("/{item_id}")
async def update_glossary_term(
    item_id: int,
    data: schemas.GlossaryTermUpdate,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> schemas.GlossaryTerm:
    """Update the received fields of a glossary term."""

    return await GlossaryOfTermsService(session).update(item_id, data)


@terms_router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_glossary_term(
    item_id: int,
    session: AsyncSession = Depends(models.get_session),
    user: User = Depends(require_jwt_auth, use_cache=False),
) -> None:
    """Delete a glossary term by id."""

    await GlossaryOfTermsService(session).delete(item_id)

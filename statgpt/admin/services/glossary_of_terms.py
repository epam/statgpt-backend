import csv
import io
import logging
import os
import zipfile
from collections import Counter
from collections.abc import Collection, Iterable
from typing import cast

from fastapi import HTTPException, status
from sqlalchemy import ColumnElement, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from statgpt.admin.settings.exim import JobsConfig
from statgpt.common import models, schemas, utils
from statgpt.common.services import ChannelService, GlossaryOfTermsService

from .exceptions import raise_for_integrity_error

_log = logging.getLogger(__name__)


def _quote_terms(terms: Iterable[str]) -> str:
    return ", ".join(repr(term) for term in sorted(terms))


def _terms_conflict_detail(terms: Collection[str | None] = ()) -> str:
    """Build a 409 detail that names the colliding term(s), so callers need not bisect."""
    named = {term for term in terms if term}
    if not named:
        return "A glossary term with the same name already exists in this channel."
    if len(named) == 1:
        return f"Key term={named.pop()!r} already exists in this channel."
    return f"Glossary terms already exist in this channel: {_quote_terms(named)}."


class AdminPortalGlossaryOfTermsService(GlossaryOfTermsService):

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, None)  # No need for session lock in Admin Portal

    @staticmethod
    def _dedupe_by_name(
        terms: list[schemas.GlossaryTermBase],
    ) -> list[schemas.GlossaryTermBase]:
        """Collapse rows that share a term name, keeping the last occurrence.

        A term name is unique per channel (uq_glossary_terms_channel_id_term), yet
        archives exported before that constraint can legitimately contain duplicate
        names (issue #564). Collapsing here keeps a bulk insert from aborting on
        the constraint. Exports are ordered by id, so the last occurrence is the most
        recently inserted row - the same one the migration keeps via MAX(id).
        """
        deduped_by_name: dict[str, schemas.GlossaryTermBase] = {}
        for term in terms:
            deduped_by_name[term.term] = term

        deduped = list(deduped_by_name.values())
        collapsed = len(terms) - len(deduped)
        if collapsed:
            dropped = _quote_terms(
                name for name, count in Counter(item.term for item in terms).items() if count > 1
            )
            _log.warning(
                f"Collapsed {collapsed} duplicate glossary term(s) sharing a name, "
                f"keeping the most recent definition of: {dropped}."
            )
        return deduped

    async def _raise_for_conflicting_terms(
        self, channel_id: int, data: list[schemas.GlossaryTermBase]
    ) -> None:
        """Reject a bulk insert up-front, naming every name that would collide.

        A term name is unique per channel (uq_glossary_terms_channel_id_term). Checking
        before the insert lets the caller see *which* rows are at fault, instead of
        getting one opaque constraint violation for the whole batch.
        """
        names = [item.term for item in data]
        if not names:
            return

        repeated = [name for name, count in Counter(names).items() if count > 1]
        if repeated:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"The request repeats glossary term(s): {_quote_terms(repeated)}.",
            )

        query = select(models.GlossaryTerm.term).where(
            models.GlossaryTerm.channel_id == channel_id,
            models.GlossaryTerm.term.in_(names),
        )
        async with self._lock_session() as session:
            conflicting = (await session.execute(query)).scalars().all()

        if conflicting:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=_terms_conflict_detail(conflicting),
            )

    async def add_term(
        self, channel_id: int, data: schemas.GlossaryTermBase
    ) -> schemas.GlossaryTerm:
        # Get channel to check if we add term to existing channel
        channel_service = ChannelService(self._session)
        channel = await channel_service.get_model_by_id(channel_id)

        term = models.GlossaryTerm(
            channel_id=channel.id,
            term=data.term,
            definition=data.definition,
            domain=data.domain,
            source=data.source,
        )
        self._session.add(term)
        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            raise_for_integrity_error(e, _terms_conflict_detail([data.term]))

        return schemas.GlossaryTerm.model_validate(term, from_attributes=True)

    async def update(self, item_id: int, data: schemas.GlossaryTermUpdate) -> schemas.GlossaryTerm:
        item = await self._get_item_or_raise(item_id)

        query = (
            update(models.GlossaryTerm)
            .where(models.GlossaryTerm.id == item.id)
            .values(**data.model_dump(mode="json", exclude_unset=True), updated_at=func.now())
            .returning(models.GlossaryTerm)
        )
        try:
            item = (await self._session.execute(query)).scalar_one()
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            raise_for_integrity_error(e, _terms_conflict_detail([data.term]))

        return schemas.GlossaryTerm.model_validate(item, from_attributes=True)

    async def delete(self, item_id: int) -> None:
        item = await self._get_item_or_raise(item_id)
        _log.info(f"Deleting {item}")

        await self._session.delete(item)
        await self._session.commit()

    async def add_terms_bulk(
        self,
        channel_id: int,
        data: list[schemas.GlossaryTermBase],
    ) -> list[schemas.GlossaryTerm]:
        channel_service = ChannelService(self._session)
        channel = await channel_service.get_model_by_id(channel_id)

        await self._raise_for_conflicting_terms(channel.id, data)

        terms = [
            models.GlossaryTerm(
                channel_id=channel.id,
                term=item.term,
                definition=item.definition,
                domain=item.domain,
                source=item.source,
            )
            for item in data
        ]

        self._session.add_all(terms)
        try:
            await self._session.commit()
        except IntegrityError as e:
            await self._session.rollback()
            # The pre-check above names the conflict in the common case; getting here
            # means a concurrent writer inserted the same name in the meantime.
            raise_for_integrity_error(e, _terms_conflict_detail())

        return [schemas.GlossaryTerm.model_validate(item, from_attributes=True) for item in terms]

    async def update_terms_bulk(
        self,
        data: list[schemas.GlossaryTermUpdateBulk],
    ) -> list[schemas.GlossaryTerm]:

        existing_terms_ids = [item.id for item in data if item.id is not None]
        existing_terms = await self._get_term_models_by_ids(existing_terms_ids)
        existing_terms_dict = {item.id: item for item in existing_terms}

        updated_item_ids: list[int] = []
        for term in data:
            if existing_term := existing_terms_dict.get(term.id):
                will_be_updated = False
                for attr, value in term.model_dump(exclude={"id"}, exclude_unset=True).items():
                    if getattr(existing_term, attr) != value:
                        setattr(existing_term, attr, value)
                        will_be_updated = True

                if will_be_updated:
                    updated_item_ids.append(cast(int, existing_term.id))
                    existing_term.updated_at = func.now()
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Term with id {term.id} not found.",
                )

        if updated_item_ids:
            _log.info(f"Updating {len(updated_item_ids)} of {len(data)} terms: {updated_item_ids}")
            try:
                await self._session.commit()
            except IntegrityError as e:
                await self._session.rollback()
                # Only a renamed term can collide, so report the names being written.
                raise_for_integrity_error(e, _terms_conflict_detail([item.term for item in data]))

            # `session.refresh()` can only be applied to one element, so it's better to query all update elements.
            existing_terms = [item for item in existing_terms if item.id not in updated_item_ids]
            existing_terms.extend(await self._get_term_models_by_ids(updated_item_ids))
        else:
            _log.info(f"All {len(data)} terms are up-to-date.")

        return [
            schemas.GlossaryTerm.model_validate(item, from_attributes=True)
            for item in existing_terms
        ]

    async def delete_terms_bulk(
        self, term_ids: list[int] | None = None, channel_id: int | None = None
    ) -> list[schemas.GlossaryTerm]:
        if term_ids is not None and channel_id is not None:
            # We can implement this feature if needed.
            raise RuntimeError("Only one of term_ids or channel_id must be provided.")

        where_clause: ColumnElement[bool]
        if term_ids is not None:
            where_clause = models.GlossaryTerm.id.in_(term_ids)
        elif channel_id is not None:
            where_clause = models.GlossaryTerm.channel_id == channel_id
        else:
            raise RuntimeError("Either term_ids or channel_id must be provided.")

        query = delete(models.GlossaryTerm).where(where_clause).returning(models.GlossaryTerm)
        deleted_terms = (await self._session.execute(query)).scalars().all()
        await self._session.commit()
        _log.info(f"Deleted {len(deleted_terms)} terms: {deleted_terms}")
        return [
            schemas.GlossaryTerm.model_validate(item, from_attributes=True)
            for item in deleted_terms
        ]

    async def export_glossary_to_folder(self, channel: models.Channel, folder_path: str) -> None:
        glossary_terms = await self.get_term_models_by_channel(channel.id, limit=None, offset=0)

        if not glossary_terms:
            _log.warning("No glossary terms found.")
            return

        _log.info(f"Exporting {len(glossary_terms)} glossary terms.")
        glossary_terms_base = [
            schemas.GlossaryTermBase.model_validate(item, from_attributes=True)
            for item in glossary_terms
        ]

        glossary_terms_data = [item.model_dump(mode="json") for item in glossary_terms_base]

        glossary_file = os.path.join(folder_path, JobsConfig.GLOSSARY_TERMS_FILE)
        utils.write_csv_from_dict_list(glossary_terms_data, glossary_file)
        _log.info(f"Exported glossary terms to {glossary_file!r}.")

    async def import_glossary_from_zip(
        self, zip_file: zipfile.ZipFile, channel_id: int, merge: bool = False
    ) -> None:
        if JobsConfig.GLOSSARY_TERMS_FILE not in zip_file.namelist():
            _log.info("No glossary terms found in the zip file.")
            return

        _log.info("Importing glossary terms from zip file.")
        with zip_file.open(JobsConfig.GLOSSARY_TERMS_FILE) as file:
            reader = csv.DictReader(io.TextIOWrapper(file, encoding='utf-8', newline=''))
            glossary_terms_data = [row for row in reader]

        glossary_terms_base = [
            schemas.GlossaryTermBase.model_validate(item) for item in glossary_terms_data
        ]

        if merge:
            await self._merge_terms(channel_id, glossary_terms_base)
            return

        # A pre-fix archive can carry duplicate names (issue #564); collapse them
        # so the insert does not abort on uq_glossary_terms_channel_id_term.
        deduped = self._dedupe_by_name(glossary_terms_base)
        items = await self.add_terms_bulk(channel_id=channel_id, data=deduped)
        _log.info(f"Imported {len(items)} glossary terms.")

    async def _merge_terms(self, channel_id: int, terms: list[schemas.GlossaryTermBase]) -> None:
        """Merge terms into an existing channel without creating duplicates.

        A term is identified by its name, which is unique per channel (see the
        ``channel_id + term`` constraint). An incoming row whose name already
        exists updates the stored fields when any differ; a new name is inserted;
        names that are unchanged (or repeated within the archive itself) are
        ignored, so re-importing the same archive stays idempotent (issue #564).
        """
        existing_by_term = {
            item.term: item
            for item in await self.get_term_models_by_channel(channel_id, limit=None, offset=0)
        }

        to_add: list[schemas.GlossaryTermBase] = []
        to_update: list[schemas.GlossaryTermUpdateBulk] = []

        for term in self._dedupe_by_name(terms):
            existing_term = existing_by_term.get(term.term)
            if existing_term is None:
                to_add.append(term)
            elif (
                existing_term.definition != term.definition
                or existing_term.domain != term.domain
                or existing_term.source != term.source
            ):
                to_update.append(
                    schemas.GlossaryTermUpdateBulk(
                        id=existing_term.id,
                        definition=term.definition,
                        domain=term.domain,
                        source=term.source,
                    )
                )

        if to_add:
            await self.add_terms_bulk(channel_id=channel_id, data=to_add)
        if to_update:
            await self.update_terms_bulk(data=to_update)

        _log.info(f"Merged glossary terms: {len(to_add)} added, {len(to_update)} updated.")

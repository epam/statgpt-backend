"""Publishing a channel's reference-area vocabulary into its own Generic RAG channel.

The chat-time pre-filter has to turn "unemployment in Germany" into a value the discovery
channel actually holds, and a channel holds several hundred area labels in whatever vocabulary
its submitters used. Offering all of them to a model at once is both expensive and imprecise,
so the labels are published as documents and searched: the query narrows the list, the model
picks from what the search returned.

One document per distinct label, whose body is the label itself and whose metadata names the
roles the channel's records use it in. Derived from the same records the discovery publisher
publishes, in the same run, right after it - so the vocabulary can only be as stale as the
documents it describes.

Nothing here is per-record. The vocabulary is a *set*: two records naming Germany produce one
document, and deleting one of them changes nothing. That is what makes this a reconciliation
against the whole record set rather than an upsert per record.
"""

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from statgpt.admin.settings.discovery import DiscoveryPublishSettings
from statgpt.common import models, schemas
from statgpt.common.services import GenericRagIngestionClient, normalize_key_part
from statgpt.common.utils import (
    MediaTypes,
    async_utils,
    escape_invalid_filename_chars,
    parse_reference_areas,
)

from .discovery_publisher import PublishCounts, first_leaf
from .exceptions import DiscoveryMetadataSchemaError, DiscoveryReferenceAreaIndexingError

_log = logging.getLogger(__name__)

_SETTINGS = DiscoveryPublishSettings()

_MAX_FILENAME_STEM = 100
_FALLBACK_FILENAME_STEM = "reference-area"
_FILENAME_DIGEST_CHARS = 12


@dataclass(frozen=True)
class _WantedArea:
    """One label the vocabulary should hold, and the roles this channel's records give it."""

    value: str
    """The spelling to publish it under."""

    roles: tuple[str, ...]
    """Sorted, so a run over unchanged records produces an unchanged document."""


def area_roles(reference_area: str) -> list[tuple[str, schemas.ReferenceAreaRole]]:
    """Every reference area one cell names, paired with the role it is named in.

    Both roles land in one vocabulary because the vocabulary answers "which of this channel's
    labels does the query mean", which is a question about spelling. The role rides along
    because the two chat-time axes filter on different document fields: a label no record ever
    names as a partner is not a value the partner axis may narrow by.
    """
    areas, partner_areas = parse_reference_areas(reference_area)
    return [
        *((value, schemas.ReferenceAreaRole.SUBJECT) for value in areas),
        *((value, schemas.ReferenceAreaRole.PARTNER) for value in partner_areas),
    ]


def document_filename(value: str, channel: str) -> str:
    """Name the uploaded file, which the channel echoes back as the document's display name.

    Same reasoning as `discovery_publisher.document_filename`: the service derives a document's
    storage path from the name alone, so the name has to identify the label on its own, and a
    readable label cannot - it drops the channel, escapes path characters to the same `_`, and
    is capped in length. The digest carries the identity, the label is there to be read.
    """
    digest = hashlib.sha256(
        "\0".join([schemas.REFERENCE_AREA_KIND, channel, normalize_key_part(value)]).encode()
    ).hexdigest()[:_FILENAME_DIGEST_CHARS]
    stem = escape_invalid_filename_chars(value)[:_MAX_FILENAME_STEM].strip()
    return f"{stem or _FALLBACK_FILENAME_STEM} [{digest}].txt"


def _build_metadata(area: _WantedArea, channel: str) -> schemas.ReferenceAreaDocumentMetadata:
    return schemas.ReferenceAreaDocumentMetadata(
        statgpt_channel=channel, value=area.value, roles=list(area.roles)
    )


def is_channel_document(document: schemas.GenericRagDocument, channel: str) -> bool:
    """Whether this document is a reference-area label published for this channel.

    Filtered on `kind` as well as on the channel so the vocabulary channel can be shared -
    by several StatGPT channels, and by whatever other kind of vocabulary comes next.
    """
    metadata = document.metadata
    return (
        metadata.get("kind") == schemas.REFERENCE_AREA_KIND
        and metadata.get("statgpt_channel") == channel
    )


def document_value(document: schemas.GenericRagDocument) -> str | None:
    """The label a document claims, or `None` if it claims none."""
    value = document.metadata.get("value")
    return value if isinstance(value, str) and value.strip() else None


def document_roles(document: schemas.GenericRagDocument) -> tuple[str, ...]:
    """The roles a document claims, in the same shape `_WantedArea` holds them.

    A document from before roles existed claims none, which differs from every wanted entry and
    is therefore republished - which is exactly the migration this needs.
    """
    roles = document.metadata.get("roles")
    if not isinstance(roles, list):
        return ()
    return tuple(sorted({role for role in roles if isinstance(role, str)}))


class ReferenceAreaPublisher:
    """Reconciles a channel's reference-area vocabulary with the labels its records name.

    One instance per run. Touches no record: a label belongs to the channel as a whole, so
    there is no per-record status to write and no per-record failure to record - either the
    vocabulary matches the records or the run reports that it does not.
    """

    def __init__(
        self,
        client: GenericRagIngestionClient,
        channel: str,
        *,
        force: bool = False,
        concurrency: int | None = None,
    ) -> None:
        """`force` rebuilds every document, whatever the channel already holds."""
        self._client = client
        self._channel = channel
        self._force = force
        self._concurrency = concurrency if concurrency is not None else _SETTINGS.concurrency

    async def verify_metadata_schema(self) -> None:
        """Refuse to publish into a channel that cannot filter on what the lookup needs."""
        declared = (await self._client.get_metadata_schema()).filterable_fields
        missing = sorted(schemas.ReferenceAreaDocumentMetadata.filterable_fields() - declared)
        if missing:
            raise DiscoveryMetadataSchemaError(missing, channel="reference-area RAG")

    async def publish(self, records: Sequence[models.DiscoveryDataset]) -> PublishCounts:
        """Bring the vocabulary channel in line with the labels `records` name.

        Only valid records contribute. An invalid record is not published, so a label only it
        names describes nothing a search could return - and offering it to a model would invite
        a filter value the discovery channel does not hold.
        """
        wanted = self._wanted_areas(records)
        counts = PublishCounts()

        existing, orphans = self._load_documents(
            [
                document
                for document in await self._client.list_documents()
                if is_channel_document(document, self._channel)
            ]
        )

        for key, document in list(existing.items()):
            if (
                key not in wanted
                or self._force
                or document.is_failed
                or document_roles(document) != wanted[key].roles
            ):
                orphans.append(existing.pop(key))

        counts.skipped = len(existing)
        published = await async_utils.gather_with_concurrency(
            self._concurrency,
            *(self._upload(area) for key, area in wanted.items() if key not in existing),
        )
        failed = sorted(value for value, document in published if document.is_failed)
        counts.upserted = len(published) - len(failed)
        counts.failed = len(failed)
        counts.deleted = await self._delete_all(orphans)

        if failed:
            raise DiscoveryReferenceAreaIndexingError(failed)

        return counts

    async def clear(self) -> int:
        """Delete this channel's whole vocabulary, returning how many documents went.

        What clearing a channel's records needs: with no records left there is nothing to
        derive, so reconciliation has nothing to reconcile against.
        """
        documents = [
            document
            for document in await self._client.list_documents()
            if is_channel_document(document, self._channel)
        ]
        deleted = await self._delete_all(documents, swallow=False)
        _log.info(f"Cleared {deleted} reference-area document(s) of channel {self._channel}")
        return deleted

    @staticmethod
    def _wanted_areas(records: Sequence[models.DiscoveryDataset]) -> dict[str, _WantedArea]:
        """Folded label -> the label the vocabulary should hold under that key.

        Folded the way the natural key is, so `EURO AREA` and `Euro area` are one document. The
        first spelling encountered wins: a channel that spells a label two ways has to pick one
        for the array fields to agree with the vocabulary, and neither spelling is more right.

        Roles are unioned rather than won: a label one record names as a subject and another as
        a partner is one document serving both axes.
        """
        spellings: dict[str, str] = {}
        roles: dict[str, set[str]] = {}
        for record in records:
            if record.validation_status is not schemas.DiscoveryValidationStatus.VALID:
                continue
            for value, role in area_roles(record.reference_area):
                key = normalize_key_part(value)
                spellings.setdefault(key, value)
                roles.setdefault(key, set()).add(role)

        return {
            key: _WantedArea(value=value, roles=tuple(sorted(roles[key])))
            for key, value in spellings.items()
        }

    @staticmethod
    def _load_documents(
        documents: Iterable[schemas.GenericRagDocument],
    ) -> tuple[dict[str, schemas.GenericRagDocument], list[schemas.GenericRagDocument]]:
        """Index this channel's vocabulary by folded label.

        Returns the index and what cannot be indexed into it: a document claiming no label, or
        a duplicate of one already seen. Both belong to no label and are deleted, keeping the
        highest id of a duplicate so the survivor is the most recently uploaded one.
        """
        by_value: dict[str, schemas.GenericRagDocument] = {}
        unclaimed: list[schemas.GenericRagDocument] = []

        for document in documents:
            value = document_value(document)
            if value is None:
                unclaimed.append(document)
                continue
            key = normalize_key_part(value)
            previous = by_value.get(key)
            if previous is None:
                by_value[key] = document
            elif document.id > previous.id:
                by_value[key] = document
                unclaimed.append(previous)
            else:
                unclaimed.append(document)

        return by_value, unclaimed

    async def _upload(self, area: _WantedArea) -> tuple[str, schemas.GenericRagDocument]:
        """Publish one label. A failure propagates and fails the run."""
        document = await self._client.upload_document(
            filename=document_filename(area.value, self._channel),
            content=area.value.encode("utf-8"),
            mime_type=MediaTypes.PLAIN_TEXT,
            metadata=_build_metadata(area, self._channel),
            # The name is derived from the label and the channel, so whatever sits under it is
            # this label's and nothing else is entitled to it - which is also what lets a label
            # whose roles changed be republished over the document it already has.
            overwrite=True,
        )
        return area.value, document

    async def _delete_all(
        self, documents: Sequence[schemas.GenericRagDocument], *, swallow: bool = True
    ) -> int:
        """Delete documents no label claims, reporting how many went.

        A failure is logged and left for the next run by default: a leftover label only widens
        the list a model is offered, and grounding drops a value the discovery channel does not
        hold. `swallow=False` is for clearing, where the caller has to know it did not finish.
        """
        try:
            deleted = await async_utils.gather_with_concurrency(
                self._concurrency,
                *(self._delete(document, swallow=swallow) for document in documents),
            )
        except BaseExceptionGroup as group:
            # Only reachable with `swallow=False`. The group is an artifact of the task group
            # inside `gather_with_concurrency`, and its leaves are one outage counted once per
            # document, so the caller is given the outage itself.
            raise first_leaf(group) from group
        return sum(deleted)

    async def _delete(self, document: schemas.GenericRagDocument, *, swallow: bool) -> int:
        try:
            await self._client.delete_document(document.id)
            return 1
        except Exception:
            if not swallow:
                raise
            _log.exception(
                f"Failed to delete reference-area document {document.id}"
                f" ({document.display_name!r}) of channel {self._channel}"
            )
            return 0

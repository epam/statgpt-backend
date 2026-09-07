"""Tests for publishing a channel's reference-area vocabulary.

The vocabulary is a set derived from every record, not a document per record, so what matters
here is set arithmetic: one document per distinct label however many records name it, carrying
every role its records use it in, a label no record names any more removed, and another
channel's labels never touched.
"""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from statgpt.admin.services.discovery_area_publisher import (
    ReferenceAreaPublisher,
    area_roles,
    document_filename,
)
from statgpt.admin.services.discovery_upload import COLUMN_FIELDS
from statgpt.admin.services.exceptions import (
    DiscoveryMetadataSchemaError,
    DiscoveryReferenceAreaIndexingError,
)
from statgpt.common import models
from statgpt.common.schemas import (
    REFERENCE_AREA_KIND,
    DiscoveryValidationStatus,
    GenericRagDocument,
    GenericRagMetadataSchema,
    ReferenceAreaDocumentMetadata,
    ReferenceAreaRole,
)
from statgpt.common.services import GenericRagIngestionClient

_CHANNEL = "statgpt-gtdc"


def _record(
    reference_area: str,
    *,
    item_id: int = 1,
    status: DiscoveryValidationStatus = DiscoveryValidationStatus.VALID,
) -> models.DiscoveryDataset:
    values: dict[str, object] = {name: "" for name in COLUMN_FIELDS}
    values.update(id=item_id, reference_area=reference_area, validation_status=status)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


def _document(
    value: str,
    document_id: int = 10,
    *,
    channel: str = _CHANNEL,
    kind: str = REFERENCE_AREA_KIND,
    status: str = "ready",
    roles: tuple[ReferenceAreaRole, ...] = (ReferenceAreaRole.SUBJECT,),
) -> GenericRagDocument:
    return GenericRagDocument(
        id=document_id,
        display_name=document_filename(value, channel),
        status=status,
        metadata={
            "kind": kind,
            "statgpt_channel": channel,
            "value": value,
            "roles": sorted(roles),
        },
    )


def _client(documents: list[GenericRagDocument] | None = None) -> AsyncMock:
    client = AsyncMock(spec=GenericRagIngestionClient)
    client.list_documents.return_value = list(documents or [])
    client.upload_document.side_effect = lambda **kwargs: GenericRagDocument(
        id=99, display_name=kwargs["filename"], status="ready"
    )
    client.get_metadata_schema.return_value = GenericRagMetadataSchema(
        schema=ReferenceAreaDocumentMetadata.channel_json_schema(), dimensions={}
    )
    return client


def _publisher(client: AsyncMock, force: bool = False) -> ReferenceAreaPublisher:
    return ReferenceAreaPublisher(
        cast(GenericRagIngestionClient, client), channel=_CHANNEL, force=force
    )


def _uploaded(client: AsyncMock) -> list[str]:
    """The labels this run published, as their metadata carries them."""
    return sorted(call.kwargs["metadata"].value for call in client.upload_document.await_args_list)


def _uploaded_roles(client: AsyncMock) -> dict[str, list[str]]:
    """The roles this run published each label under."""
    return {
        call.kwargs["metadata"].value: call.kwargs["metadata"].roles
        for call in client.upload_document.await_args_list
    }


def _deleted(client: AsyncMock) -> list[int]:
    return sorted(call.args[0] for call in client.delete_document.await_args_list)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ what a record contributes ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_a_cell_contributes_each_label_under_the_role_it_names_it_in() -> None:
    """One vocabulary, two roles: which label a query means, and which axis may filter on it."""
    assert area_roles("France; partner countries: China") == [
        ("France", ReferenceAreaRole.SUBJECT),
        ("China", ReferenceAreaRole.PARTNER),
    ]


def _digest(filename: str) -> str:
    return filename.rsplit("[", 1)[1]


def test_one_label_is_one_document_however_it_is_spelled() -> None:
    """The digest carries the identity, so the readable part is free to differ."""
    assert _digest(document_filename("Euro area", _CHANNEL)) == _digest(
        document_filename("euro  area", _CHANNEL)
    )


def test_two_channels_never_share_a_document() -> None:
    """The service derives a document's storage path from its name alone."""
    assert document_filename("Euro area", _CHANNEL) != document_filename("Euro area", "other")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the metadata schema guard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_channel_that_cannot_filter_is_refused() -> None:
    client = _client()
    schema = ReferenceAreaDocumentMetadata.channel_json_schema()
    del schema["properties"]["statgpt_channel"]["enable_filtering"]
    client.get_metadata_schema.return_value = GenericRagMetadataSchema(schema=schema, dimensions={})

    with pytest.raises(DiscoveryMetadataSchemaError, match="statgpt_channel"):
        await _publisher(client).verify_metadata_schema()


async def test_a_channel_declaring_the_required_filters_is_accepted() -> None:
    await _publisher(_client()).verify_metadata_schema()


def test_the_roles_render_as_a_plain_string_array() -> None:
    """The only shape of array the service can turn into a request model.

    An optional or enum-typed array makes that derivation raise on every search of the channel,
    not only one filtering on roles - so this guards the whole vocabulary's searchability.
    """
    properties = ReferenceAreaDocumentMetadata.channel_json_schema()["properties"]

    assert properties["roles"] == {
        "type": "array",
        "items": {"type": "string"},
        "enable_filtering": True,
    }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ reconciliation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_labels_are_deduplicated_across_records_and_carry_every_role() -> None:
    client = _client()
    records = [
        _record("France; Germany", item_id=1),
        _record("germany; partner countries: France; China", item_id=2),
    ]

    counts = await _publisher(client).publish(records)

    assert _uploaded_roles(client) == {
        "France": ["partner", "subject"],
        "Germany": ["subject"],
        "China": ["partner"],
    }
    assert counts.upserted == 3
    assert counts.deleted == 0


async def test_a_label_no_record_names_any_more_is_deleted() -> None:
    client = _client([_document("France", 10), _document("Atlantis", 11)])

    counts = await _publisher(client).publish([_record("France")])

    assert _uploaded(client) == []
    assert _deleted(client) == [11]
    assert (counts.upserted, counts.skipped, counts.deleted) == (0, 1, 1)


async def test_only_valid_records_contribute() -> None:
    """An invalid record is not published, so its labels describe nothing a search can return."""
    client = _client()
    records = [
        _record("France"),
        _record("Atlantis", item_id=2, status=DiscoveryValidationStatus.INVALID),
    ]

    await _publisher(client).publish(records)

    assert _uploaded(client) == ["France"]


async def test_another_channels_documents_are_left_alone() -> None:
    """One vocabulary channel can serve several StatGPT channels."""
    client = _client(
        [
            _document("Atlantis", 10, channel="other-channel"),
            _document("Atlantis", 11, kind="something-else"),
        ]
    )

    counts = await _publisher(client).publish([_record("France")])

    assert _uploaded(client) == ["France"]
    assert _deleted(client) == []
    assert counts.deleted == 0


async def test_a_document_claiming_no_label_is_swept() -> None:
    client = _client(
        [
            GenericRagDocument(
                id=12, metadata={"kind": REFERENCE_AREA_KIND, "statgpt_channel": _CHANNEL}
            )
        ]
    )

    await _publisher(client).publish([_record("France")])

    assert _deleted(client) == [12]


async def test_a_duplicated_label_keeps_the_most_recent_document() -> None:
    client = _client([_document("France", 10), _document("france", 11)])

    counts = await _publisher(client).publish([_record("France")])

    assert _deleted(client) == [10]
    assert _uploaded(client) == []
    assert counts.skipped == 1


async def test_a_failed_document_is_rebuilt() -> None:
    """A document in `error` holds no content, so the label it stands for is unreachable."""
    client = _client([_document("France", 10, status="error")])

    await _publisher(client).publish([_record("France")])

    assert _deleted(client) == [10]
    assert _uploaded(client) == ["France"]


async def test_a_label_whose_roles_changed_is_republished() -> None:
    """The document says which axis may filter on the label, so a drifted one misleads.

    It is also the migration: a document from before roles existed claims none, differs from
    every wanted entry, and is rebuilt by the next run.
    """
    client = _client([_document("France", 10), _document("China", 11)])
    del client.list_documents.return_value[1].metadata["roles"]
    records = [
        _record("France", item_id=1),
        _record("partner countries: France; China", item_id=2),
    ]

    counts = await _publisher(client).publish(records)

    assert _uploaded_roles(client) == {
        "France": ["partner", "subject"],
        "China": ["partner"],
    }
    assert _deleted(client) == [10, 11]
    assert counts.skipped == 0


async def test_a_forced_run_rebuilds_every_document() -> None:
    client = _client([_document("France", 10)])

    await _publisher(client, force=True).publish([_record("France")])

    assert _deleted(client) == [10]
    assert _uploaded(client) == ["France"]


async def test_an_unpublishable_label_fails_the_run() -> None:
    """A missing label narrows every query naming it away from datasets that do cover it."""
    client = _client()
    client.upload_document.side_effect = lambda **kwargs: GenericRagDocument(
        id=99, display_name=kwargs["filename"], status="error"
    )

    with pytest.raises(DiscoveryReferenceAreaIndexingError, match="France"):
        await _publisher(client).publish([_record("France")])


async def test_clearing_removes_only_this_channels_labels() -> None:
    client = _client([_document("France", 10), _document("Atlantis", 11, channel="other")])

    assert await _publisher(client).clear() == 1
    assert _deleted(client) == [10]


async def test_a_failure_while_clearing_propagates() -> None:
    """The caller is about to drop the records that are the last thing pointing at them."""
    client = _client([_document("France", 10)])
    client.delete_document.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await _publisher(client).clear()

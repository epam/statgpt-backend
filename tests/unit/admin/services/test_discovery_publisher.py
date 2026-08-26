"""Tests for publishing discovery dataset records into a Generic RAG channel."""

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from statgpt.admin.services import discovery_publisher as publisher_module
from statgpt.admin.services.discovery_publisher import (
    DiscoveryPublisher,
    build_metadata,
    document_filename,
    document_key,
    render_document_body,
)
from statgpt.admin.services.discovery_upload import COLUMN_FIELDS
from statgpt.admin.services.exceptions import DiscoveryMetadataSchemaError
from statgpt.common import models
from statgpt.common.schemas import (
    DiscoveryDocumentMetadata,
    DiscoveryGrade,
    DiscoveryIndexingStatus,
    DiscoveryValidationStatus,
    GenericRagDocument,
    GenericRagMetadataSchema,
)
from statgpt.common.services import GenericRagIngestionClient
from statgpt.common.services.generic_rag.ingestion import GenericRagIngestionError

_CHANNEL = "statgpt-gtdc"

_FULL_RECORD = {
    "reference_area": "Indonesia (IDN)",
    "regional_coverage": "None",
    "excluded_regional_values": "",
    "agency": "Bank Indonesia (BI)",
    "dataset_id": "TABEL1_1",
    "name": "I.1. Broad Money and its Affecting Factors",
    "description": "Money and Banking table.",
    "url": "https://www.bi.go.id/SEKI/tabel/TABEL1_1.xls",
    "time_coverage": "From 1989-01 to 2026-06",
    "frequency_coverage": "Monthly",
    "indicators_coverage": "broad money (M2) (Rp billions)",
    "missing_indicators": "policy interest rates",
}


def _record(**overrides: object) -> models.DiscoveryDataset:
    """A stand-in for a stored row, carrying only the columns the publisher touches."""
    values: dict[str, object] = {name: "" for name in COLUMN_FIELDS}
    values.update(_FULL_RECORD)
    values.update(
        id=1,
        channel_id=1,
        validation_status=DiscoveryValidationStatus.VALID,
        validation_issues=None,
        indexing_status=DiscoveryIndexingStatus.NEW,
        indexed_at=None,
        index_error=None,
    )
    values.update(overrides)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


def _document(
    document_id: int = 10,
    agency: str = "Bank Indonesia (BI)",
    dataset_id: str = "TABEL1_1",
    grade: str = DiscoveryGrade.C,
    channel: str = _CHANNEL,
    status: str = "ready",
    display_name: str | None = None,
) -> GenericRagDocument:
    """A document as the channel would return it.

    Its display name defaults to the one the publisher would generate for the matching
    record, since that is what decides whether a refresh can happen in place.
    """
    return GenericRagDocument(
        id=document_id,
        display_name=(
            display_name
            if display_name is not None
            else document_filename(_record(agency=agency, dataset_id=dataset_id), channel, grade)
        ),
        status=status,
        metadata={
            "grade": grade,
            "statgpt_channel": channel,
            "agency": agency,
            "dataset_id": dataset_id,
        },
    )


def _client(documents: list[GenericRagDocument] | None = None) -> AsyncMock:
    client = AsyncMock(spec=GenericRagIngestionClient)
    client.list_documents.return_value = list(documents or [])
    client.upload_document.return_value = _document(document_id=99)
    client.update_document.side_effect = lambda document_id, **_: _document(document_id=document_id)
    client.get_metadata_schema.return_value = GenericRagMetadataSchema(
        schema=DiscoveryDocumentMetadata.channel_json_schema(), dimensions={}
    )
    return client


def _publisher(client: AsyncMock, force: bool = False) -> DiscoveryPublisher:
    return DiscoveryPublisher(
        cast(GenericRagIngestionClient, client),
        channel=_CHANNEL,
        grade=DiscoveryGrade.C,
        force=force,
        settle_interval_seconds=0,
    )


def _deleted_ids(client: AsyncMock) -> list[int]:
    """Sorted, because the records are published concurrently."""
    return sorted(call.args[0] for call in client.delete_document.call_args_list)


def _updated_ids(client: AsyncMock) -> list[int]:
    return sorted(call.args[0] for call in client.update_document.call_args_list)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the document ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_the_body_carries_every_field_including_those_sent_as_metadata() -> None:
    """Metadata is what search filters on; the body is what it retrieves over."""
    body = render_document_body(_record())

    assert body.startswith("# I.1. Broad Money and its Affecting Factors\n")
    assert "- **Agency / organization:** Bank Indonesia (BI)" in body
    assert "- **Dataset URL:** https://www.bi.go.id/SEKI/tabel/TABEL1_1.xls" in body
    assert "## Description\n\nMoney and Banking table." in body
    assert "## Indicators coverage\n\nbroad money (M2) (Rp billions)" in body
    assert body.endswith("\n")


def test_the_body_omits_empty_fields() -> None:
    body = render_document_body(_record(missing_indicators="", regional_coverage=""))

    assert "Relevant indicators not present" not in body
    assert "Regional coverage" not in body


def test_the_body_falls_back_to_the_dataset_id_when_unnamed() -> None:
    assert render_document_body(_record(name="")).startswith("# TABEL1_1\n")


def _filename(record, channel: str = _CHANNEL) -> str:
    return document_filename(record, channel, DiscoveryGrade.C)


def test_the_filename_stays_readable() -> None:
    assert _filename(_record()).startswith("Bank Indonesia (BI) - TABEL1_1 [")


def test_the_filename_escapes_path_characters() -> None:
    name = _filename(_record(agency="A/B", dataset_id="C:D"))

    assert "/" not in name
    assert name.endswith(".md")


def test_the_readable_part_of_the_filename_is_capped() -> None:
    name = _filename(_record(agency="A" * 200))

    assert name.startswith("A" * 100 + " [")
    assert name.endswith(".md")


def test_the_filename_is_stable_for_one_record() -> None:
    assert _filename(_record()) == _filename(_record())


def test_respelling_a_record_only_changes_the_readable_label() -> None:
    """Identity lives in the digest; a record re-spelled is still the same document."""
    respelled = _filename(_record(agency="  BANK  Indonesia (BI) ", dataset_id="tabel1_1"))
    original = _filename(_record())

    assert respelled != original
    assert respelled[-16:] == original[-16:]  # ' [<digest>].md'


# The service derives a document's storage path from its name alone, so any two records that
# would be named the same become one document and one of them vanishes from the index.


def test_the_same_record_in_two_channels_gets_two_filenames() -> None:
    record = _record()

    assert _filename(record, "statgpt-gtdc") != _filename(record, "statgpt-other")


def test_the_same_record_in_two_grades_gets_two_filenames() -> None:
    record = _record()

    assert document_filename(record, _CHANNEL, DiscoveryGrade.B) != document_filename(
        record, _CHANNEL, DiscoveryGrade.C
    )


def test_ids_that_escape_to_the_same_label_get_two_filenames() -> None:
    assert _filename(_record(dataset_id="DF/1.0")) != _filename(_record(dataset_id="DF_1.0"))


def test_records_sharing_a_truncated_label_get_two_filenames() -> None:
    agency = "N" * 120

    assert _filename(_record(agency=agency, dataset_id="GDP_QUARTERLY")) != _filename(
        _record(agency=agency, dataset_id="CPI_MONTHLY")
    )


def test_the_metadata_carries_every_field_except_the_description() -> None:
    metadata = build_metadata(_record(), _CHANNEL, DiscoveryGrade.C)

    dumped = metadata.model_dump()
    assert dumped["grade"] == "C"
    assert dumped["statgpt_channel"] == _CHANNEL
    for field, value in _FULL_RECORD.items():
        if field == "description":
            assert field not in dumped
        else:
            assert dumped[field] == value


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ matching documents to records ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_a_documents_key_is_folded_like_the_natural_key() -> None:
    document = _document(agency="  BANK  Indonesia (BI) ", dataset_id="tabel1_1")

    assert document_key(document) == ("bank indonesia (bi)", "tabel1_1")


@pytest.mark.parametrize(
    "metadata",
    [{}, {"agency": "BI"}, {"agency": "BI", "dataset_id": ""}, {"agency": 7, "dataset_id": "X"}],
)
def test_a_document_without_a_usable_key_claims_none(metadata: dict[str, object]) -> None:
    document = GenericRagDocument(id=1, metadata=metadata)

    assert document_key(document) is None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the metadata schema guard ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_channel_that_cannot_filter_is_refused() -> None:
    client = _client()
    schema = DiscoveryDocumentMetadata.channel_json_schema()
    del schema["properties"]["reference_area_values"]["enable_filtering"]
    client.get_metadata_schema.return_value = GenericRagMetadataSchema(schema=schema, dimensions={})

    with pytest.raises(DiscoveryMetadataSchemaError, match="reference_area_values"):
        await _publisher(client).verify_metadata_schema()


async def test_the_generated_schema_declares_everything_a_run_requires() -> None:
    """One definition: the schema an administrator configures and the check are the same."""
    declared = GenericRagMetadataSchema(
        schema=DiscoveryDocumentMetadata.channel_json_schema(), dimensions={}
    ).filterable_fields

    assert declared == DiscoveryDocumentMetadata.filterable_fields()
    assert {"agency", "reference_area_values", "grade", "statgpt_channel"} <= declared


async def test_the_country_axis_is_filterable_and_the_verbatim_cell_is_not() -> None:
    """`reference_area` is free text - one country, a ';'-separated list, or a group label - so
    equality against the whole cell never matches a question about one member of a multi-country
    dataset. `reference_area_values` is the axis a filter can match, and the cell is kept for
    display only."""
    filterable = DiscoveryDocumentMetadata.filterable_fields()

    assert "reference_area_values" in filterable
    assert "reference_area" not in filterable


async def test_a_channel_declaring_the_required_filters_is_accepted() -> None:
    await _publisher(_client()).verify_metadata_schema()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ publishing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_new_record_is_uploaded_and_marked_indexed() -> None:
    client = _client()
    record = _record()

    counts = await _publisher(client).publish([record])

    assert client.upload_document.await_count == 1
    assert client.delete_document.await_count == 0
    assert counts.upserted == 1
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED
    assert record.indexed_at is not None
    assert record.index_error is None


async def test_an_edited_record_is_refreshed_in_place() -> None:
    """Its document keeps its id, and is never briefly missing from the channel."""
    client = _client([_document(document_id=10)])
    record = _record(indexing_status=DiscoveryIndexingStatus.OUTDATED)

    counts = await _publisher(client).publish([record])

    assert _updated_ids(client) == [10]
    assert client.delete_document.await_count == 0
    assert client.upload_document.await_count == 0
    assert (counts.upserted, counts.deleted, counts.skipped) == (1, 0, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED


async def test_a_record_whose_label_changed_is_rebuilt_under_the_new_name() -> None:
    """An update cannot rename, so the document would keep the old label for good."""
    client = _client([_document(document_id=10, display_name="Bank Indonesia (BI) - OLD.md")])
    record = _record(indexing_status=DiscoveryIndexingStatus.OUTDATED)

    counts = await _publisher(client).publish([record])

    assert _deleted_ids(client) == [10]
    assert client.upload_document.await_count == 1
    assert client.update_document.await_count == 0
    assert (counts.upserted, counts.deleted) == (1, 1)


async def test_an_indexed_record_with_a_healthy_document_is_skipped() -> None:
    client = _client([_document()])
    record = _record(indexing_status=DiscoveryIndexingStatus.INDEXED)

    counts = await _publisher(client).publish([record])

    assert client.upload_document.await_count == 0
    assert client.delete_document.await_count == 0
    assert counts.skipped == 1
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED


async def test_an_indexed_record_whose_document_failed_is_rebuilt_not_refreshed() -> None:
    """A refresh sends the same bytes, the etag matches, and the parse is never retried."""
    client = _client([_document(status="error")])
    record = _record(indexing_status=DiscoveryIndexingStatus.INDEXED)

    counts = await _publisher(client).publish([record])

    assert client.update_document.await_count == 0
    assert _deleted_ids(client) == [10]
    assert client.upload_document.await_count == 1
    assert (counts.upserted, counts.deleted) == (1, 1)
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED


async def test_an_indexed_record_whose_document_vanished_is_published_again() -> None:
    client = _client([])
    record = _record(indexing_status=DiscoveryIndexingStatus.INDEXED)

    counts = await _publisher(client).publish([record])

    assert client.upload_document.await_count == 1
    assert counts.upserted == 1


async def test_a_record_matches_its_document_however_the_key_was_cased() -> None:
    client = _client([_document(agency="bank indonesia (bi)", dataset_id="TABEL1_1")])
    record = _record(indexing_status=DiscoveryIndexingStatus.INDEXED)

    counts = await _publisher(client).publish([record])

    assert (counts.skipped, counts.deleted) == (1, 0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ withdrawal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_an_invalid_record_has_its_document_withdrawn_and_goes_back_to_new() -> None:
    client = _client([_document(document_id=10)])
    record = _record(
        validation_status=DiscoveryValidationStatus.INVALID,
        indexing_status=DiscoveryIndexingStatus.INDEXED,
        indexed_at="yesterday",
    )

    counts = await _publisher(client).publish([record])

    assert _deleted_ids(client) == [10]
    assert client.upload_document.await_count == 0
    assert (counts.deleted, counts.upserted) == (1, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.NEW
    assert record.indexed_at is None
    assert record.index_error is None


async def test_an_invalid_record_that_was_never_published_needs_no_request() -> None:
    client = _client()
    record = _record(validation_status=DiscoveryValidationStatus.INVALID)

    counts = await _publisher(client).publish([record])

    assert client.delete_document.await_count == 0
    assert client.upload_document.await_count == 0
    assert (counts.deleted, counts.upserted, counts.failed) == (0, 0, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.NEW


async def test_the_document_is_deleted_before_the_status_is_written() -> None:
    """A crash in between must leave a claim to be indexed, not a stranded document."""
    client = _client([_document()])
    record = _record(
        validation_status=DiscoveryValidationStatus.INVALID,
        indexing_status=DiscoveryIndexingStatus.INDEXED,
    )
    client.delete_document.side_effect = GenericRagIngestionError("document deletion", "boom", 503)

    counts = await _publisher(client).publish([record])

    assert counts.failed == 1
    assert record.indexing_status is DiscoveryIndexingStatus.FAILED
    assert "document deletion" in (record.index_error or "")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ reconciliation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_document_no_record_claims_is_deleted() -> None:
    client = _client([_document(document_id=10, dataset_id="TABEL9_9")])

    counts = await _publisher(client).publish([_record()])

    assert _deleted_ids(client) == [10]
    assert counts.deleted == 1


async def test_a_document_claiming_no_key_is_deleted() -> None:
    orphan = GenericRagDocument(id=11, metadata={"grade": "C", "statgpt_channel": _CHANNEL})
    client = _client([orphan])

    counts = await _publisher(client).publish([])

    assert _deleted_ids(client) == [11]
    assert counts.deleted == 1


async def test_duplicates_of_one_key_keep_the_newest_document() -> None:
    client = _client([_document(document_id=10), _document(document_id=12)])
    record = _record(indexing_status=DiscoveryIndexingStatus.INDEXED)

    counts = await _publisher(client).publish([record])

    assert _deleted_ids(client) == [10]
    assert (counts.skipped, counts.deleted) == (1, 1)


async def test_documents_of_other_channels_and_grades_are_left_alone() -> None:
    client = _client(
        [
            _document(document_id=10, channel="statgpt-other"),
            _document(document_id=11, grade=DiscoveryGrade.B),
            _document(document_id=12, dataset_id="TABEL9_9"),
        ]
    )

    counts = await _publisher(client).publish([])

    assert _deleted_ids(client) == [12]
    assert counts.deleted == 1


async def test_an_orphan_that_cannot_be_deleted_does_not_fail_the_run() -> None:
    """It is a retrieval nuisance for the next run to clear, not a publishing failure."""
    client = _client([_document(document_id=10, dataset_id="TABEL9_9")])
    client.delete_document.side_effect = GenericRagIngestionError("document deletion", "boom", 503)

    counts = await _publisher(client).publish([_record()])

    assert counts.upserted == 1
    assert (counts.deleted, counts.failed) == (0, 0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ failure isolation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_one_failing_record_does_not_stop_the_others() -> None:
    client = _client()
    first, second = _record(id=1), _record(id=2, dataset_id="TABEL2_2")
    client.upload_document.side_effect = [
        GenericRagIngestionError("document upload", "bad metadata", 422),
        _document(document_id=99),
    ]

    counts = await _publisher(client).publish([first, second])

    assert (counts.upserted, counts.failed) == (1, 1)
    assert first.indexing_status is DiscoveryIndexingStatus.FAILED
    assert "document upload" in (first.index_error or "")
    assert second.indexing_status is DiscoveryIndexingStatus.INDEXED


async def test_a_failed_upload_after_a_successful_delete_leaves_nothing_indexed() -> None:
    client = _client([_document(document_id=10, display_name="renamed.md")])
    record = _record(indexing_status=DiscoveryIndexingStatus.OUTDATED)
    client.upload_document.side_effect = GenericRagIngestionError("document upload", "boom", 503)

    counts = await _publisher(client).publish([record])

    assert _deleted_ids(client) == [10]
    assert (counts.upserted, counts.failed, counts.deleted) == (0, 1, 1)
    assert record.indexing_status is DiscoveryIndexingStatus.FAILED


async def test_a_failed_refresh_is_recorded_on_the_record() -> None:
    client = _client([_document(document_id=10)])
    record = _record(indexing_status=DiscoveryIndexingStatus.OUTDATED)
    client.update_document.side_effect = GenericRagIngestionError("document update", "boom", 503)

    counts = await _publisher(client).publish([record])

    assert (counts.upserted, counts.failed, counts.deleted) == (0, 1, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.FAILED
    assert "document update" in (record.index_error or "")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ forced rebuilds ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_a_forced_run_republishes_a_record_it_would_have_skipped() -> None:
    """The only way to pick up a change no record's status records."""
    client = _client([_document(document_id=10)])
    record = _record(indexing_status=DiscoveryIndexingStatus.INDEXED)

    counts = await _publisher(client, force=True).publish([record])

    assert (counts.upserted, counts.skipped) == (1, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED


async def test_a_forced_run_rebuilds_rather_than_refreshes() -> None:
    """An update would send identical bytes, and the channel would re-index nothing."""
    client = _client([_document(document_id=10)])
    record = _record(indexing_status=DiscoveryIndexingStatus.INDEXED)

    counts = await _publisher(client, force=True).publish([record])

    assert _deleted_ids(client) == [10]
    assert client.upload_document.await_count == 1
    assert client.update_document.await_count == 0
    assert counts.deleted == 1


async def test_a_forced_run_still_withdraws_an_invalid_record() -> None:
    client = _client([_document(document_id=10)])
    record = _record(
        validation_status=DiscoveryValidationStatus.INVALID,
        indexing_status=DiscoveryIndexingStatus.INDEXED,
    )

    counts = await _publisher(client, force=True).publish([record])

    assert _deleted_ids(client) == [10]
    assert (counts.upserted, counts.deleted) == (0, 1)
    assert record.indexing_status is DiscoveryIndexingStatus.NEW


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ settling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# A channel that indexes in the background answers the upload long before the document is
# retrievable, so an upload response saying '201' is not evidence that anything was indexed.


def _settling_client(*passes: list[GenericRagDocument]) -> AsyncMock:
    """A channel whose upload leaves the document mid-flight.

    Each argument is what `list_documents` returns on one settle pass; the first listing,
    which happens before anything is published, always finds the channel empty.

    The last pass then repeats for as long as it is asked for. A finite list would raise
    once exhausted, `_settle` would swallow that as a failed confirmation, and a test
    expecting the deadline to be reached would pass without ever reaching it.
    """
    client = _client()
    client.upload_document.return_value = _document(document_id=99, status="created")
    client.list_documents.side_effect = _listings([], *passes)
    return client


def _listings(*passes: list[GenericRagDocument]) -> Callable[[], list[GenericRagDocument]]:
    """Answer each listing from `passes` in turn, repeating the last one forever."""
    remaining = list(passes)

    def next_listing() -> list[GenericRagDocument]:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return next_listing


async def test_a_document_that_ends_in_error_fails_its_record_in_the_same_run() -> None:
    client = _settling_client([_document(document_id=99, status="error")])
    record = _record()

    counts = await _publisher(client).publish([record])

    assert (counts.upserted, counts.failed) == (0, 1)
    assert record.indexing_status is DiscoveryIndexingStatus.FAILED
    assert "parse or index" in (record.index_error or "")


async def test_a_settled_failure_keeps_the_time_the_record_was_uploaded() -> None:
    """It says when the record was last uploaded, which the failure does not undo."""
    client = _settling_client([_document(document_id=99, status="error")])
    record = _record()

    await _publisher(client).publish([record])

    assert record.indexed_at is not None


async def test_a_document_that_reaches_ready_leaves_its_record_indexed() -> None:
    client = _settling_client(
        [_document(document_id=99, status="indexing")],
        [_document(document_id=99, status="ready")],
    )
    record = _record()

    counts = await _publisher(client).publish([record])

    assert (counts.upserted, counts.failed) == (1, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED
    assert client.list_documents.await_count == 3


async def test_a_document_still_indexing_at_the_deadline_is_left_to_the_next_run() -> None:
    client = _settling_client([_document(document_id=99, status="processing")])
    record = _record()

    publisher = DiscoveryPublisher(
        cast(GenericRagIngestionClient, client),
        channel=_CHANNEL,
        settle_timeout_seconds=0.0001,
        settle_interval_seconds=0,
    )
    counts = await publisher.publish([record])

    assert (counts.upserted, counts.failed) == (1, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED
    # The listing never stops reporting the document as in flight, so the loop can only have
    # ended by reaching the deadline.
    assert client.list_documents.await_count >= 2


async def test_settling_is_skipped_when_the_channel_indexes_before_it_answers() -> None:
    """Nothing to wait for, so not one extra request."""
    client = _client()

    await _publisher(client).publish([_record()])

    assert client.list_documents.await_count == 1


async def test_a_listing_that_fails_while_settling_leaves_the_publish_results_alone() -> None:
    client = _settling_client()
    client.list_documents.side_effect = [
        [],
        GenericRagIngestionError("document listing", "boom", 503),
    ]
    record = _record()

    counts = await _publisher(client).publish([record])

    assert (counts.upserted, counts.failed) == (1, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED


async def test_a_document_that_disappears_while_settling_is_left_to_the_next_run() -> None:
    client = _settling_client([])
    record = _record()

    counts = await _publisher(client).publish([record])

    assert (counts.upserted, counts.failed) == (1, 0)
    assert record.indexing_status is DiscoveryIndexingStatus.INDEXED


async def test_the_settle_interval_backs_off_to_its_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quick at first, for the common case, then patient rather than chatty."""
    slept: list[float] = []

    async def record_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(publisher_module.asyncio, "sleep", record_sleep)
    client = _settling_client()
    client.list_documents.side_effect = _listings(
        [],
        *[[_document(document_id=99, status="indexing")]] * 5,
        [_document(document_id=99, status="ready")],
    )

    publisher = DiscoveryPublisher(
        cast(GenericRagIngestionClient, client), channel=_CHANNEL, grade=DiscoveryGrade.C
    )
    counts = await publisher.publish([_record()])

    assert counts.upserted == 1
    assert slept == [1.0, 2.0, 4.0, 5.0, 5.0]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ concurrency ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def test_records_are_published_concurrently() -> None:
    in_flight = peak = 0

    async def upload(**_: object) -> GenericRagDocument:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return _document(document_id=99)

    client = _client()
    client.upload_document.side_effect = upload
    records = [_record(id=i, dataset_id=f"TABEL{i}") for i in range(6)]

    counts = await _publisher(client).publish(records)

    assert counts.upserted == 6
    assert peak > 1


async def test_concurrency_is_bounded() -> None:
    in_flight = peak = 0

    async def upload(**_: object) -> GenericRagDocument:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return _document(document_id=99)

    client = _client()
    client.upload_document.side_effect = upload
    records = [_record(id=i, dataset_id=f"TABEL{i}") for i in range(10)]

    publisher = DiscoveryPublisher(
        cast(GenericRagIngestionClient, client), channel=_CHANNEL, concurrency=2
    )
    await publisher.publish(records)

    assert peak <= 2


async def test_one_records_failure_does_not_cancel_the_others() -> None:
    """The turns share a task group, so an escaping exception would take the run down."""
    client = _client()
    records = [_record(id=i, dataset_id=f"TABEL{i}") for i in range(5)]

    async def upload(*, filename: str, **_: object) -> GenericRagDocument:
        if "TABEL2" in filename:
            raise GenericRagIngestionError("document upload", "boom", 503)
        return _document(document_id=99)

    client.upload_document.side_effect = upload

    counts = await _publisher(client).publish(records)

    assert (counts.upserted, counts.failed) == (4, 1)
    assert [r.indexing_status is DiscoveryIndexingStatus.INDEXED for r in records] == [
        True,
        True,
        False,
        True,
        True,
    ]

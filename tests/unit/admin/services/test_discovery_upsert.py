from collections.abc import Sequence
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from statgpt.admin.services.discovery_dataset import (
    AdminPortalDiscoveryDatasetService,
    _build_candidates,
    _build_candidates_from_rows,
    _DiscoveryCandidate,
    _duplicate_problems,
    _raise_for_problems,
    _RawRecord,
)
from statgpt.admin.services.discovery_upload import COLUMN_FIELDS
from statgpt.admin.services.exceptions import DiscoveryPayloadError
from statgpt.common import models
from statgpt.common.schemas import (
    DiscoveryDatasetBase,
    DiscoveryDatasetUpdate,
    DiscoveryDatasetUpdateBulk,
    DiscoveryIndexingStatus,
    DiscoveryUploadMode,
    DiscoveryUploadSummary,
    DiscoveryValidationStatus,
)
from statgpt.common.services import DiscoveryDatasetService, normalize_key_part, record_key

_MAX_PROBLEMS = 200


def _record(
    agency: str = "Bank Indonesia (BI)", dataset_id: str = "TABEL1_1", **overrides: str
) -> DiscoveryDatasetBase:
    values: dict[str, str] = {"agency": agency, "dataset_id": dataset_id}
    values.update(overrides)
    return DiscoveryDatasetBase.model_validate(values)


def _stored(
    agency: str = "Bank Indonesia (BI)", dataset_id: str = "TABEL1_1", **overrides: object
) -> models.DiscoveryDataset:
    """A stand-in for a stored row: the columns the service reads and writes.

    Cast rather than built, so the service's own signatures still type-check at the call
    sites below; constructing a real mapped instance would need a session.
    """
    values: dict[str, object] = {name: "" for name in COLUMN_FIELDS}
    values.update(agency=agency, dataset_id=dataset_id)
    values.update(overrides)
    values.setdefault("id", 1)
    values.setdefault("validation_status", DiscoveryValidationStatus.VALID)
    values.setdefault("validation_issues", [{"field": "url", "message": "old"}])
    values.setdefault("validated_at", "yesterday")
    values.setdefault("indexing_status", DiscoveryIndexingStatus.INDEXED)
    values.setdefault("index_error", None)
    values.setdefault("updated_at", None)
    return cast(models.DiscoveryDataset, SimpleNamespace(**values))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ the natural key ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.mark.parametrize(
    "agency",
    [
        "Bank Indonesia (BI)",
        "bank indonesia (bi)",
        "  Bank  Indonesia (BI) ",
        "BANK INDONESIA (BI)",
    ],
)
def test_key_ignores_case_and_spacing(agency: str) -> None:
    """Whitespace is normalized by the schema, case is folded by the key."""
    record = _record(agency=agency)

    assert record_key(record.agency, record.dataset_id) == ("bank indonesia (bi)", "tabel1_1")


def test_non_breaking_space_normalizes_to_the_same_key() -> None:
    """Cells pasted from web pages routinely carry U+00A0, and a key built from one would
    be a duplicate record."""
    record = _record(agency="Bank Indonesia (BI)")

    assert record_key(record.agency, record.dataset_id) == ("bank indonesia (bi)", "tabel1_1")


def test_stored_value_keeps_its_submitted_casing() -> None:
    """Case is folded into a derived key, never into the value shown in the UI."""
    assert _record(agency="IMF").agency == "IMF"


@pytest.mark.parametrize("field_name", ["agency", "dataset_id"])
@pytest.mark.parametrize("value", ["", "   ", " "])
def test_an_empty_key_field_is_refused_by_the_schema(field_name: str, value: str) -> None:
    """Enforced on the model, so no write path can forget it.

    Whitespace-only counts as empty: the check runs after normalization, which is why the
    order of the annotated validators matters.
    """
    with pytest.raises(ValidationError) as exc_info:
        _record(**{field_name: value})

    assert exc_info.value.errors()[0]["loc"] == (field_name,)


@pytest.mark.parametrize("field_name", ["agency", "dataset_id"])
def test_an_edit_may_correct_the_key_but_not_clear_it(field_name: str) -> None:
    assert DiscoveryDatasetUpdate.model_validate({field_name: "IMF"})

    with pytest.raises(ValidationError):
        DiscoveryDatasetUpdate.model_validate({field_name: ""})


@pytest.mark.parametrize(
    "submitted",
    [
        "Bank Indonesia (BI)",
        "bank indonesia (bi)",
        "  Bank  Indonesia (BI) ",
        "Bank\xa0Indonesia (BI)",
    ],
)
def test_the_agency_filter_folds_exactly_like_the_stored_key(submitted: str) -> None:
    """The read filter and the write path have to agree, or a list request silently returns
    nothing for a value the UI itself displays."""
    stored_key = record_key("  Bank  Indonesia (BI) ", "TABEL1_1")[0]

    clause = DiscoveryDatasetService._filters(channel_id=1, agency=submitted)[-1]
    rendered = str(clause.compile(compile_kwargs={"literal_binds": True}))

    assert normalize_key_part(submitted) == stored_key
    assert f"'{stored_key}'" in rendered


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ structural validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_a_complete_payload_passes() -> None:
    candidates = _build_candidates([_record(), _record(dataset_id="TABEL1_2")])

    assert _duplicate_problems(candidates) == []


def test_duplicate_keys_within_one_payload_are_rejected_naming_both_rows() -> None:
    """Comparing lowercased keys here is what keeps this a 400 instead of an IntegrityError."""
    payload = [_record(agency="IMF"), _record(agency="imf ")]

    with pytest.raises(DiscoveryPayloadError) as exc_info:
        _raise_for_problems(_duplicate_problems(_build_candidates(payload)), _MAX_PROBLEMS)

    problem = exc_info.value.problems[0]
    assert problem.index == 1
    assert "Duplicate of item 0" in problem.message


def test_problem_list_is_capped_and_marked_truncated() -> None:
    payload = [_record() for _ in range(6)]  # five duplicates of item 0

    with pytest.raises(DiscoveryPayloadError) as exc_info:
        _raise_for_problems(_duplicate_problems(_build_candidates(payload)), max_problems=2)

    assert len(exc_info.value.problems) == 2
    assert exc_info.value.truncated is True
    assert exc_info.value.detail.truncated is True


def test_file_problems_carry_row_and_cell_references() -> None:
    candidate = _DiscoveryCandidate(record=_record(), row=14, cells={"dataset_id": "E14"})
    duplicate = _DiscoveryCandidate(record=_record(), row=15, cells={"dataset_id": "E15"})

    problems = _duplicate_problems([candidate, duplicate])

    assert [(p.row, p.cell, p.index) for p in problems] == [(15, "E15", None)]


@pytest.mark.parametrize(
    "field_name, cell, label",
    [
        ("agency", "D14", "Agency / organization"),
        # `dataset_id` aliases to `datasetId`, and pydantic locates the error by alias - so
        # this is the case that catches the alias mapping going missing.
        ("dataset_id", "E14", "Dataset ID"),
    ],
)
def test_an_empty_key_in_a_file_is_reported_against_its_cell(
    field_name: str, cell: str, label: str
) -> None:
    """A schema violation has to arrive as a located problem, not as a pydantic dump."""
    values = {"agency": "Bank Indonesia (BI)", "dataset_id": "TABEL1_1"}
    values[field_name] = "   "
    raw = [
        _RawRecord(values=values, row=14, cells={"agency": "D14", "dataset_id": "E14"}),
    ]

    candidates, problems = _build_candidates_from_rows(raw)

    assert candidates == []
    assert len(problems) == 1
    assert (problems[0].field, problems[0].row, problems[0].cell) == (field_name, 14, cell)
    assert problems[0].message == f"{label} must not be empty."


def test_a_row_missing_a_key_column_is_reported_as_missing() -> None:
    candidates, problems = _build_candidates_from_rows(
        [_RawRecord(values={"agency": "IMF"}, index=3)]
    )

    assert candidates == []
    assert [(p.field, p.index, p.message) for p in problems] == [
        ("dataset_id", 3, "Dataset ID is missing.")
    ]


def test_the_exception_message_names_the_offending_records() -> None:
    """`str(exc)` is what reaches a log line and an import job's `reason_for_failure`."""
    _, problems = _build_candidates_from_rows(
        [_RawRecord(values={"agency": "", "dataset_id": "X"}, row=14, cells={"agency": "D14"})]
    )

    with pytest.raises(DiscoveryPayloadError) as exc_info:
        _raise_for_problems(problems, _MAX_PROBLEMS)

    assert "D14: Agency / organization must not be empty." in str(exc_info.value)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ upsert reconciliation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _service(existing: list[models.DiscoveryDataset]) -> AdminPortalDiscoveryDatasetService:
    service = AdminPortalDiscoveryDatasetService(session=MagicMock())
    service._session.commit = AsyncMock()  # type: ignore[method-assign]
    service._session.execute = AsyncMock()  # type: ignore[method-assign]
    service.get_record_models_by_channel = AsyncMock(return_value=existing)  # type: ignore[method-assign]
    return service


async def _upload(
    service: AdminPortalDiscoveryDatasetService,
    records: Sequence[DiscoveryDatasetBase],
    mode: DiscoveryUploadMode,
) -> DiscoveryUploadSummary:
    return await service._upsert(
        channel_id=1,
        candidates=_build_candidates(records),
        delete_absent=mode is DiscoveryUploadMode.REPLACE,
    )


@pytest.mark.asyncio
async def test_new_records_are_created() -> None:
    service = _service(existing=[])

    summary = await _upload(service, [_record()], DiscoveryUploadMode.UPSERT)

    assert (summary.created, summary.updated, summary.unchanged, summary.deleted) == (1, 0, 0, 0)


@pytest.mark.asyncio
async def test_identical_resubmission_is_unchanged_and_keeps_statuses() -> None:
    """An unedited re-upload must not reset a record's verdict or republish its document."""
    stored = _stored()
    service = _service(existing=[stored])

    summary = await _upload(service, [_record()], DiscoveryUploadMode.UPSERT)

    assert (summary.created, summary.updated, summary.unchanged) == (0, 0, 1)
    assert stored.validation_status is DiscoveryValidationStatus.VALID
    assert stored.indexing_status is DiscoveryIndexingStatus.INDEXED


@pytest.mark.parametrize("agency", ["bank indonesia (bi)", "  Bank  Indonesia (BI)  "])
@pytest.mark.asyncio
async def test_a_case_or_spacing_only_difference_is_unchanged(agency: str) -> None:
    """Otherwise a copy-paste in a resubmitted workbook silently republishes the record."""
    stored = _stored()
    service = _service(existing=[stored])

    summary = await _upload(service, [_record(agency=agency)], DiscoveryUploadMode.UPSERT)

    assert (summary.created, summary.updated, summary.unchanged, summary.deleted) == (0, 0, 1, 0)
    assert stored.agency == "Bank Indonesia (BI)"  # stored casing wins
    assert stored.indexing_status is DiscoveryIndexingStatus.INDEXED


@pytest.mark.asyncio
async def test_a_payload_that_omits_a_field_leaves_the_stored_value_alone() -> None:
    """A file may carry a subset of the columns, and the rest must survive.

    `model_fields_set` is the line between "the file did not mention this" and "the submitter
    cleared this cell"; comparing values instead would blank ten fields for anyone uploading
    a two-column correction.
    """
    stored = _stored(description="Money and banking table.", url="https://www.bi.go.id/x")
    service = _service(existing=[stored])

    # Exactly what a CSV holding only `agency,dataset_id` produces.
    partial = DiscoveryDatasetBase.model_validate(
        {"agency": "Bank Indonesia (BI)", "dataset_id": "TABEL1_1"}
    )

    summary = await _upload(service, [partial], DiscoveryUploadMode.UPSERT)

    assert (summary.created, summary.updated, summary.unchanged) == (0, 0, 1)
    assert stored.description == "Money and banking table."
    assert stored.url == "https://www.bi.go.id/x"
    assert stored.indexing_status is DiscoveryIndexingStatus.INDEXED


@pytest.mark.asyncio
async def test_an_omitted_field_is_not_confused_with_a_cleared_one() -> None:
    """A column the file does carry, left blank, is a deliberate clearing."""
    stored = _stored(description="Money and banking table.")
    service = _service(existing=[stored])

    summary = await _upload(service, [_record(description="")], DiscoveryUploadMode.UPSERT)

    assert (summary.created, summary.updated, summary.unchanged) == (0, 1, 0)
    assert stored.description == ""


@pytest.mark.asyncio
async def test_an_edited_field_updates_the_record_and_resets_its_state() -> None:
    stored = _stored(description="old")
    service = _service(existing=[stored])

    summary = await _upload(service, [_record(description="new")], DiscoveryUploadMode.UPSERT)

    assert (summary.created, summary.updated, summary.unchanged) == (0, 1, 0)
    assert stored.description == "new"
    assert stored.validation_status is DiscoveryValidationStatus.NOT_VALIDATED
    assert stored.validation_issues is None
    assert stored.validated_at is None
    # It had been published, so the indexed document is now stale.
    assert stored.indexing_status is DiscoveryIndexingStatus.OUTDATED


@pytest.mark.asyncio
async def test_an_edited_record_that_was_never_indexed_stays_new() -> None:
    stored = _stored(description="old", indexing_status=DiscoveryIndexingStatus.NEW)
    service = _service(existing=[stored])

    await _upload(service, [_record(description="new")], DiscoveryUploadMode.UPSERT)

    assert stored.indexing_status is DiscoveryIndexingStatus.NEW


@pytest.mark.asyncio
async def test_editing_a_failed_record_clears_the_publish_error() -> None:
    """`index_error` is set exactly while `indexing_status` is FAILED, and the old failure
    described the old content."""
    stored = _stored(
        description="old",
        indexing_status=DiscoveryIndexingStatus.FAILED,
        index_error="Publishing to the discovery RAG is not implemented yet.",
    )
    service = _service(existing=[stored])

    await _upload(service, [_record(description="new")], DiscoveryUploadMode.UPSERT)

    assert stored.indexing_status is DiscoveryIndexingStatus.NEW
    assert stored.index_error is None


@pytest.mark.asyncio
async def test_upsert_keeps_records_the_file_does_not_mention() -> None:
    stored = _stored(dataset_id="TABEL9_9")
    service = _service(existing=[stored])

    summary = await _upload(service, [_record()], DiscoveryUploadMode.UPSERT)

    assert (summary.created, summary.deleted) == (1, 0)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ edits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# `_apply_values` takes the payload model, so these cover the shape an edit arrives in:
# a subset of the fields, where `null` means "not provided" rather than "clear this".


def test_an_edit_writes_only_the_fields_it_carries() -> None:
    stored = _stored(description="old", name="Broad Money")

    changed = AdminPortalDiscoveryDatasetService._apply_values(
        stored, DiscoveryDatasetUpdate.model_validate({"description": "new"}), ignore_key_case=False
    )

    assert changed is True
    assert (stored.description, stored.name) == ("new", "Broad Money")
    assert stored.indexing_status is DiscoveryIndexingStatus.OUTDATED


def test_an_explicit_null_in_an_edit_is_not_a_cleared_field() -> None:
    """`null` is how a partial JSON body says "not provided"; only `""` clears a field."""
    stored = _stored(description="old")

    changed = AdminPortalDiscoveryDatasetService._apply_values(
        stored, DiscoveryDatasetUpdate.model_validate({"description": None}), ignore_key_case=False
    )

    assert changed is False
    assert stored.description == "old"
    assert stored.validation_status is DiscoveryValidationStatus.VALID


def test_a_bulk_edit_does_not_write_the_id_it_addresses() -> None:
    """`id` rides along in the bulk payload but is not a descriptive column."""
    stored = _stored(description="old")

    AdminPortalDiscoveryDatasetService._apply_values(
        stored,
        DiscoveryDatasetUpdateBulk.model_validate({"id": 99, "description": "new"}),
        ignore_key_case=False,
    )

    assert (stored.id, stored.description) == (1, "new")


def test_an_edit_rewrites_the_key_casing() -> None:
    """Unlike an upload, a single-record edit of the key is a deliberate correction."""
    stored = _stored(agency="Bank Indonesia (BI)")

    changed = AdminPortalDiscoveryDatasetService._apply_values(
        stored,
        DiscoveryDatasetUpdate.model_validate({"agency": "BANK INDONESIA (BI)"}),
        ignore_key_case=False,
    )

    assert changed is True
    assert stored.agency == "BANK INDONESIA (BI)"


@pytest.mark.asyncio
async def test_replace_deletes_records_the_file_does_not_mention() -> None:
    absent = _stored(dataset_id="TABEL9_9")
    kept = _stored(dataset_id="TABEL1_1", id=2)
    service = _service(existing=[absent, kept])

    summary = await _upload(service, [_record()], DiscoveryUploadMode.REPLACE)

    assert (summary.created, summary.unchanged, summary.deleted) == (0, 1, 1)

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services.discovery_dataset import (
    AdminPortalDiscoveryDatasetService,
    build_candidates,
    record_key,
    validate_candidates,
)
from statgpt.admin.services.discovery_upload import COLUMN_FIELDS
from statgpt.admin.services.exceptions import DiscoveryPayloadError
from statgpt.common.schemas import (
    DiscoveryDatasetBase,
    DiscoveryIndexingStatus,
    DiscoveryUploadMode,
    DiscoveryValidationStatus,
)

_MAX_PROBLEMS = 200


def _record(agency: str = "Bank Indonesia (BI)", dataset_id: str = "TABEL1_1", **overrides: str):
    values: dict[str, str] = {"agency": agency, "dataset_id": dataset_id}
    values.update(overrides)
    return DiscoveryDatasetBase.model_validate(values)


def _stored(agency: str = "Bank Indonesia (BI)", dataset_id: str = "TABEL1_1", **overrides):
    """A stand-in for a stored row: the columns the service reads and writes."""
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
    return SimpleNamespace(**values)


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
    """The agency below holds a literal U+00A0 between 'Bank' and 'Indonesia'.

    Cells pasted from web pages routinely carry them, and a key built from one would be a
    duplicate record.
    """
    record = _record(agency="Bank Indonesia (BI)")

    assert record_key(record.agency, record.dataset_id) == ("bank indonesia (bi)", "tabel1_1")


def test_stored_value_keeps_its_submitted_casing() -> None:
    """Case is folded into a derived key, never into the value shown in the UI."""
    assert _record(agency="IMF").agency == "IMF"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ structural validation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_a_complete_payload_passes() -> None:
    validate_candidates(
        build_candidates([_record(), _record(dataset_id="TABEL1_2")]), _MAX_PROBLEMS
    )


@pytest.mark.parametrize("field_name", ["agency", "dataset_id"])
def test_an_empty_key_field_is_rejected(field_name: str) -> None:
    payload = [_record(), _record(**{field_name: "   "})]

    with pytest.raises(DiscoveryPayloadError) as exc_info:
        validate_candidates(build_candidates(payload), _MAX_PROBLEMS)

    problems = exc_info.value.problems
    assert [problem.field for problem in problems] == [field_name]
    assert [problem.index for problem in problems] == [1]
    assert exc_info.value.detail.message.startswith("The payload has 1 problem;")


def test_duplicate_keys_within_one_payload_are_rejected_naming_both_rows() -> None:
    """Comparing lowercased keys here is what keeps this a 400 instead of an IntegrityError."""
    payload = [_record(agency="IMF"), _record(agency="imf ")]

    with pytest.raises(DiscoveryPayloadError) as exc_info:
        validate_candidates(build_candidates(payload), _MAX_PROBLEMS)

    problem = exc_info.value.problems[0]
    assert problem.index == 1
    assert "Duplicate of item 0" in problem.message


def test_problem_list_is_capped_and_marked_truncated() -> None:
    payload = [_record(agency="", dataset_id=f"D{i}") for i in range(5)]

    with pytest.raises(DiscoveryPayloadError) as exc_info:
        validate_candidates(build_candidates(payload), max_problems=2)

    assert len(exc_info.value.problems) == 2
    assert exc_info.value.truncated is True
    assert exc_info.value.detail.truncated is True


def test_a_row_with_no_key_is_not_also_reported_as_a_duplicate() -> None:
    """Two keyless rows are two empty-field problems, not an additional duplicate."""
    payload = [_record(agency=""), _record(agency="")]

    with pytest.raises(DiscoveryPayloadError) as exc_info:
        validate_candidates(build_candidates(payload), _MAX_PROBLEMS)

    assert [problem.field for problem in exc_info.value.problems] == ["agency", "agency"]


def test_file_problems_carry_row_and_cell_references() -> None:
    from statgpt.admin.services.discovery_dataset import DiscoveryCandidate

    candidate = DiscoveryCandidate(record=_record(agency=""), row=14, cells={"agency": "D14"})

    with pytest.raises(DiscoveryPayloadError) as exc_info:
        validate_candidates([candidate], _MAX_PROBLEMS)

    problem = exc_info.value.problems[0]
    assert (problem.row, problem.cell, problem.index) == (14, "D14", None)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ upsert reconciliation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _service(existing: list) -> AdminPortalDiscoveryDatasetService:
    service = AdminPortalDiscoveryDatasetService(session=MagicMock())
    service._session.commit = AsyncMock()  # type: ignore[method-assign]
    service._session.delete = AsyncMock()  # type: ignore[method-assign]
    service.get_record_models_by_channel = AsyncMock(return_value=existing)  # type: ignore[method-assign]
    return service


async def _upload(service: AdminPortalDiscoveryDatasetService, records, mode):
    return await service._upsert(
        channel_id=1,
        candidates=build_candidates(records),
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
async def test_upsert_keeps_records_the_file_does_not_mention() -> None:
    stored = _stored(dataset_id="TABEL9_9")
    service = _service(existing=[stored])

    summary = await _upload(service, [_record()], DiscoveryUploadMode.UPSERT)

    assert (summary.created, summary.deleted) == (1, 0)
    service._session.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_replace_deletes_records_the_file_does_not_mention() -> None:
    absent = _stored(dataset_id="TABEL9_9")
    kept = _stored(dataset_id="TABEL1_1", id=2)
    service = _service(existing=[absent, kept])

    summary = await _upload(service, [_record()], DiscoveryUploadMode.REPLACE)

    assert (summary.created, summary.unchanged, summary.deleted) == (0, 1, 1)
    service._session.delete.assert_awaited_once_with(absent)

import csv
import io
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from statgpt.admin.services.glossary_of_terms import AdminPortalGlossaryOfTermsService
from statgpt.admin.settings.exim import JobsConfig

_FIELDS = ["term", "definition", "domain", "source"]


def _make_zip(rows: list[dict[str, str]]) -> zipfile.ZipFile:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        archive.writestr(JobsConfig.GLOSSARY_TERMS_FILE, csv_buffer.getvalue())
    buffer.seek(0)
    return zipfile.ZipFile(buffer, "r")


def _term(
    term: str,
    definition: str = "def",
    domain: str = "Econ",
    source: str = "IMF",
    term_id: int = 1,
):
    return SimpleNamespace(
        id=term_id, term=term, definition=definition, domain=domain, source=source
    )


def _row(term: str, definition: str = "def", domain: str = "Econ", source: str = "IMF") -> dict:
    return {"term": term, "definition": definition, "domain": domain, "source": source}


def _make_service(existing: list) -> AdminPortalGlossaryOfTermsService:
    service = AdminPortalGlossaryOfTermsService(session=MagicMock())
    service.get_term_models_by_channel = AsyncMock(return_value=existing)  # type: ignore[method-assign]
    service.add_terms_bulk = AsyncMock(return_value=[])  # type: ignore[method-assign]
    service.update_terms_bulk = AsyncMock(return_value=[])  # type: ignore[method-assign]
    return service


@pytest.mark.asyncio
async def test_merge_adds_only_missing_terms() -> None:
    service = _make_service(existing=[_term("GDP")])
    zip_file = _make_zip([_row("GDP"), _row("CPI")])

    await service.import_glossary_from_zip(zip_file, channel_id=1, merge=True)

    service.add_terms_bulk.assert_awaited_once()
    added = service.add_terms_bulk.await_args.kwargs["data"]
    assert [item.term for item in added] == ["CPI"]
    service.update_terms_bulk.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_reimport_is_idempotent() -> None:
    """Re-importing the same archive into an existing channel changes nothing (issue #564)."""
    service = _make_service(existing=[_term("GDP"), _term("CPI", term_id=2)])
    zip_file = _make_zip([_row("GDP"), _row("CPI")])

    await service.import_glossary_from_zip(zip_file, channel_id=1, merge=True)

    service.add_terms_bulk.assert_not_awaited()
    service.update_terms_bulk.assert_not_awaited()


@pytest.mark.asyncio
async def test_merge_updates_edited_definition() -> None:
    """An edited definition updates the existing term instead of duplicating it (issue #564)."""
    service = _make_service(existing=[_term("GDP", definition="old", term_id=7)])
    zip_file = _make_zip([_row("GDP", definition="new")])

    await service.import_glossary_from_zip(zip_file, channel_id=1, merge=True)

    service.add_terms_bulk.assert_not_awaited()
    service.update_terms_bulk.assert_awaited_once()
    updates = service.update_terms_bulk.await_args.kwargs["data"]
    assert [(u.id, u.definition) for u in updates] == [(7, "new")]


@pytest.mark.asyncio
async def test_merge_dedupes_rows_within_archive() -> None:
    """Rows duplicated inside the archive itself collapse to a single insert (issue #564)."""
    service = _make_service(existing=[])
    zip_file = _make_zip([_row("GDP"), _row("GDP")])

    await service.import_glossary_from_zip(zip_file, channel_id=1, merge=True)

    service.add_terms_bulk.assert_awaited_once()
    added = service.add_terms_bulk.await_args.kwargs["data"]
    assert [item.term for item in added] == ["GDP"]


@pytest.mark.asyncio
async def test_merge_treats_different_domain_as_distinct() -> None:
    """Identity includes domain, so the same name under a new domain is added, not updated."""
    service = _make_service(existing=[_term("GDP", domain="Econ")])
    zip_file = _make_zip([_row("GDP", domain="Trade")])

    await service.import_glossary_from_zip(zip_file, channel_id=1, merge=True)

    service.add_terms_bulk.assert_awaited_once()
    added = service.add_terms_bulk.await_args.kwargs["data"]
    assert [(item.term, item.domain) for item in added] == [("GDP", "Trade")]
    service.update_terms_bulk.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_merge_adds_all_without_checking_existing() -> None:
    service = _make_service(existing=[_term("GDP")])
    zip_file = _make_zip([_row("GDP")])

    await service.import_glossary_from_zip(zip_file, channel_id=1, merge=False)

    service.get_term_models_by_channel.assert_not_awaited()
    service.add_terms_bulk.assert_awaited_once()
    added = service.add_terms_bulk.await_args.kwargs["data"]
    assert [item.term for item in added] == ["GDP"]
    service.update_terms_bulk.assert_not_awaited()

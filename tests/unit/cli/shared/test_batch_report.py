"""Tests for the batch outcome report shared by `content init` and `channel reindex`."""

import re

from statgpt.cli.shared.batch_report import BatchItemStatus, BatchReport

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(out: str) -> str:
    """Drop styling, so assertions read the text and not Rich's colour codes."""
    return _ANSI.sub("", out)


def _report() -> BatchReport:
    report = BatchReport(title="Summary")
    report.record_ok("dataset", "A")
    report.record_unchanged("dataset", "B")
    report.record_skipped("dataset", "C", "its data source failed")
    report.record_failed("dataset", "D", "HTTP 400: bad DSD")
    return report


def test_counts_cover_every_recorded_status() -> None:
    counts = _report().counts()

    assert counts == {
        BatchItemStatus.FAILED: 1,
        BatchItemStatus.SKIPPED: 1,
        BatchItemStatus.OK: 1,
        BatchItemStatus.UNCHANGED: 1,
    }


def test_counts_omit_statuses_that_did_not_occur() -> None:
    report = BatchReport(title="Summary")
    report.record_ok("dataset", "A")

    assert report.counts() == {BatchItemStatus.OK: 1}


def test_failed_and_skipped_are_reported_separately() -> None:
    report = _report()

    assert [item.item_id for item in report.failed] == ["D"]
    assert [item.item_id for item in report.skipped] == ["C"]


def test_has_failures_is_true_only_for_failures() -> None:
    assert _report().has_failures

    skipped_only = BatchReport(title="Summary")
    skipped_only.record_skipped("dataset", "C", "its data source failed")
    assert not skipped_only.has_failures

    assert not BatchReport(title="Summary").has_failures


def test_render_lists_failures_before_skips_and_omits_successes(capsys) -> None:
    """The table is for items needing action; successes are already logged inline."""
    _report().render()

    out = _plain(capsys.readouterr().out)
    assert out.index("FAILED") < out.index("SKIPPED")
    # "A" and "B" succeeded, so they get no row - only the counts mention them.
    assert "HTTP 400: bad DSD" in out
    assert "its data source failed" in out
    assert "1 failed, 1 skipped, 1 ok, 1 unchanged" in out


def test_render_escapes_markup_in_error_bodies(capsys) -> None:
    """A JSON `detail` list is square-bracketed; Rich must not read it as markup."""
    report = BatchReport(title="Summary")
    report.record_failed("dataset", "D", '[{"error": "bad DSD"}]')

    report.render()

    assert '[{"error": "bad DSD"}]' in _plain(capsys.readouterr().out)


def test_render_truncates_a_very_long_message(capsys) -> None:
    report = BatchReport(title="Summary")
    report.record_failed("dataset", "D", "x" * 5000)

    report.render()

    out = _plain(capsys.readouterr().out)
    assert "…" in out
    assert "x" * 400 not in out


def test_render_handles_an_empty_report(capsys) -> None:
    BatchReport(title="Summary").render()

    assert "Nothing to do." in _plain(capsys.readouterr().out)

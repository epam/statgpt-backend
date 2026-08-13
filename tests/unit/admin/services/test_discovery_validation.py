import pytest

from statgpt.admin.services.discovery_validation import (
    DEFAULT_CHECKS,
    FREQUENCY_VOCABULARY,
    DiscoveryCheck,
    DiscoveryRecord,
    DiscoveryValidator,
)
from statgpt.common.schemas import DiscoveryDatasetBase, DiscoveryValidationIssue


def _record(**overrides: str) -> DiscoveryDatasetBase:
    values: dict[str, str] = {
        "agency": "Bank Indonesia (BI)",
        "dataset_id": "TABEL1_1",
        "frequency_coverage": "Monthly",
        "url": "https://www.bi.go.id/SEKI/tabel/TABEL1_1.xls",
    }
    values.update(overrides)
    return DiscoveryDatasetBase.model_validate(values)


def test_a_well_filled_record_is_valid() -> None:
    assert DiscoveryValidator().validate(_record()) == []


@pytest.mark.parametrize("frequency", FREQUENCY_VOCABULARY)
def test_every_vocabulary_frequency_is_accepted(frequency: str) -> None:
    assert DiscoveryValidator().validate(_record(frequency_coverage=frequency)) == []


def test_frequencies_are_matched_per_token_and_case_insensitively() -> None:
    record = _record(frequency_coverage="monthly; Quarterly ;ANNUAL")

    assert DiscoveryValidator().validate(record) == []


def test_a_frequency_outside_the_vocabulary_is_reported() -> None:
    issues = DiscoveryValidator().validate(_record(frequency_coverage="Monthly; Fortnightly"))

    assert len(issues) == 1
    assert issues[0].field == "frequency_coverage"
    assert "'Fortnightly'" in issues[0].message
    # The message lists what to choose from, so a submitter can fix it without the template.
    assert "Semi-annual" in issues[0].message


def test_an_empty_frequency_is_not_an_issue() -> None:
    """Absent information does not make a record unfit to refer to; a wrong value does."""
    assert DiscoveryValidator().validate(_record(frequency_coverage="")) == []


@pytest.mark.parametrize(
    "url",
    [
        "https://data.imf.org/en/datasets/IMF.STA:DOT",
        # Plain http is accepted: an invalid record is not published, and demanding https
        # would delist an agency that publishes over http.
        "http://www.example.org/data",
    ],
)
def test_web_addresses_are_accepted(url: str) -> None:
    assert DiscoveryValidator().validate(_record(url=url)) == []


@pytest.mark.parametrize(
    "url",
    ["N/A", "see our website", "ftp://data.example.org/dot", "www.example.org/data"],
)
def test_a_value_that_is_not_a_web_address_is_reported(url: str) -> None:
    issues = DiscoveryValidator().validate(_record(url=url))

    assert [issue.field for issue in issues] == ["url"]
    assert repr(url) in issues[0].message


def test_an_empty_url_is_not_an_issue() -> None:
    assert DiscoveryValidator().validate(_record(url="")) == []


def test_group_reference_areas_are_not_flagged() -> None:
    """The template explicitly allows group labels, so there is no ISO-code check."""
    assert DiscoveryValidator().validate(_record(reference_area="Euro area")) == []


def test_all_issues_are_collected_not_just_the_first() -> None:
    issues = DiscoveryValidator().validate(_record(frequency_coverage="Fortnightly", url="N/A"))

    assert sorted(issue.field for issue in issues) == ["frequency_coverage", "url"]


def test_the_check_set_is_pluggable() -> None:
    """The metadata guidelines are still being written, so the set must be replaceable."""

    def always_fails(record: DiscoveryRecord) -> list[DiscoveryValidationIssue]:
        return [DiscoveryValidationIssue(field="name", message="Nope.")]

    validator = DiscoveryValidator(checks=[DiscoveryCheck(name="custom", run=always_fails)])

    assert [issue.message for issue in validator.validate(_record())] == ["Nope."]


def test_a_broken_check_becomes_an_issue_rather_than_aborting_the_run() -> None:
    def explodes(record: DiscoveryRecord) -> list[DiscoveryValidationIssue]:
        raise RuntimeError("boom")

    checks = [DiscoveryCheck(name="explodes", run=explodes), *DEFAULT_CHECKS]

    issues = DiscoveryValidator(checks=checks).validate(_record())

    assert [issue.field for issue in issues] == ["explodes"]

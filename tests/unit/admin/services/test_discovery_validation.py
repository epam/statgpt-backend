import pytest

from statgpt.admin.services.discovery_validation import (
    DEFAULT_CHECKS,
    DiscoveryCheck,
    DiscoveryRecord,
    DiscoveryValidator,
)
from statgpt.common.schemas import DiscoveryDatasetBase, DiscoveryValidationIssue
from statgpt.common.utils import FREQUENCY_VOCABULARY


def _record(**overrides: str) -> DiscoveryDatasetBase:
    values: dict[str, str] = {
        "agency": "Bank Indonesia (BI)",
        "dataset_id": "TABEL1_1",
        "description": "Money and Banking table.",
        "reference_area": "Indonesia",
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


@pytest.mark.parametrize("frequency", ["", "  ", ";"])
def test_a_record_naming_no_frequency_is_reported(frequency: str) -> None:
    """The chat-time pre-filter narrows by frequency, so a record naming none is unreachable."""
    issues = DiscoveryValidator().validate(_record(frequency_coverage=frequency))

    assert [issue.field for issue in issues] == ["frequency_coverage"]
    # The message says what to choose from, so a submitter can fix it without the template.
    assert "Semi-annual" in issues[0].message


@pytest.mark.parametrize("reference_area", ["", "  ", ";"])
def test_a_record_naming_no_reference_area_is_reported(reference_area: str) -> None:
    """Same reason: an empty axis would be filtered out of every narrowed search."""
    issues = DiscoveryValidator().validate(_record(reference_area=reference_area))

    assert [issue.field for issue in issues] == ["reference_area"]


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


def test_a_record_without_a_description_is_reported() -> None:
    """The description is the published document's only content."""
    issues = DiscoveryValidator().validate(_record(description=""))

    assert len(issues) == 1
    assert issues[0].field == "description"


@pytest.mark.parametrize("reference_area", ["Euro area", "World", "partner countries: China"])
def test_group_and_partner_reference_areas_are_not_flagged(reference_area: str) -> None:
    """The template allows group labels and partner lists, so there is no ISO-code check."""
    assert DiscoveryValidator().validate(_record(reference_area=reference_area)) == []


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

"""Tests for rendering a discovery search result as a referral."""

from statgpt.app.chains.discovery.referral import GROUNDING_RULE, render_referral
from statgpt.app.schemas.discovery import (
    DiscoveryCandidate,
    DiscoveryReferralItem,
    DiscoverySearchResult,
)


def _item(
    name: str = "Monetary Base",
    *,
    url: str = "https://example.com/boj",
    reason: str = "",
    missing: str = "",
    **fields,
) -> DiscoveryReferralItem:
    return DiscoveryReferralItem(
        candidate=DiscoveryCandidate(
            document_id=1,
            display_name="boj.md",
            name=name,
            url=url,
            agency=fields.pop("agency", "Bank of Japan (BOJ)"),
            **fields,
        ),
        reason=reason,
        missing=missing,
    )


def test_an_empty_result_renders_nothing():
    """The tool appends the result unconditionally, so "nothing to refer to" has to be empty
    rather than an empty heading."""
    assert render_referral(DiscoverySearchResult()) == ""


def test_a_referral_names_the_dataset_and_links_its_official_source():
    """The link is the point of the whole feature."""
    rendered = render_referral(DiscoverySearchResult(items=[_item()]))

    assert "**Monetary Base**" in rendered
    assert "https://example.com/boj" in rendered
    assert "Bank of Japan (BOJ)" in rendered


def test_a_referral_says_the_datasets_cannot_be_queried_here():
    rendered = render_referral(DiscoverySearchResult(items=[_item()]))

    assert "cannot be queried" in rendered


def test_a_referral_carries_the_rule_that_bounds_the_agent():
    """A record's indicator list reads exactly like something the agent could answer from, so the
    rule travels with the data rather than living only in the system prompt."""
    rendered = render_referral(DiscoverySearchResult(items=[_item()]))

    assert GROUNDING_RULE in rendered
    assert "not data" in rendered


def test_coverage_details_are_rendered_when_present():
    rendered = render_referral(
        DiscoverySearchResult(
            items=[
                _item(
                    reference_area="Japan (JPN)",
                    time_coverage="From 1970-01 to present",
                    frequency_coverage="Monthly; Annual",
                )
            ]
        )
    )

    assert "Coverage: Japan (JPN)" in rendered
    assert "Period: From 1970-01 to present" in rendered
    assert "Frequency: Monthly; Annual" in rendered


def test_absent_details_are_omitted_rather_than_rendered_empty():
    rendered = render_referral(DiscoverySearchResult(items=[_item(reference_area="")]))

    assert "Coverage:" not in rendered
    assert "Period:" not in rendered


def test_the_judges_reason_is_shown():
    rendered = render_referral(DiscoverySearchResult(items=[_item(reason="publishes M2 monthly")]))

    assert "Why: publishes M2 monthly" in rendered


def test_what_the_dataset_does_not_cover_is_stated_rather_than_dropped():
    """Telling the user an official dataset exists but excludes part of what they asked saves
    them a click, and it is what the record's own text says."""
    rendered = render_referral(
        DiscoverySearchResult(items=[_item(missing="services trade is not covered")])
    )

    assert "Not covered: services trade is not covered" in rendered


def test_a_dataset_with_no_url_still_renders_without_a_source_line():
    """A record that failed validation should never reach here, but a missing URL must not raise."""
    rendered = render_referral(DiscoverySearchResult(items=[_item(url="")]))

    assert "**Monetary Base**" in rendered
    assert "Source:" not in rendered


def test_every_selected_dataset_appears_in_order():
    result = DiscoverySearchResult(items=[_item("First"), _item("Second"), _item("Third")])

    rendered = render_referral(result)

    assert rendered.index("First") < rendered.index("Second") < rendered.index("Third")


def test_the_label_falls_back_to_the_dataset_id_then_the_document_name():
    """A record whose name cell was left empty still has to be nameable in a referral."""
    unnamed = DiscoveryReferralItem(
        candidate=DiscoveryCandidate(
            document_id=1, display_name="boj.md", name="", dataset_id="TEST_BOJ_MB"
        )
    )

    assert "TEST_BOJ_MB" in render_referral(DiscoverySearchResult(items=[unnamed]))

"""Tests for rendering the discovery datasets a relevance judge kept."""

from statgpt.app.chains.discovery_datasets.templates import render_block, render_item
from statgpt.app.schemas.discovery_datasets import DiscoveryCandidate
from statgpt.common.schemas import DiscoveryDocumentMetadata
from statgpt.common.schemas.discovery_datasets_tool import DiscoveryDatasetsTemplates


def _candidate(document_id: int = 1, rank: int = 1, **metadata: str) -> DiscoveryCandidate:
    fields: dict[str, str] = {
        "grade": "C",
        "statgpt_channel": "statgpt-gtdc",
        "agency": "IMF",
        "name": f"Dataset {document_id}",
    }
    fields.update(metadata)
    return DiscoveryCandidate(
        document_id=document_id,
        rank=rank,
        display_name=f"doc{document_id}.txt",
        metadata=DiscoveryDocumentMetadata(**fields),
        description=f"Description {document_id}.",
    )


def _templates(item: str, wrapper: str = "### Datasets\n\n{items}") -> DiscoveryDatasetsTemplates:
    return DiscoveryDatasetsTemplates(wrapper=wrapper, item=item)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ items ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_an_item_can_use_metadata_and_retrieval_placeholders() -> None:
    candidate = _candidate(3, rank=2, url="https://data.imf.org/x", time_coverage="1995-2024")

    rendered = render_item(
        "{rank}. {name} ({agency}) {url} [{time_coverage}] {description} #{document_id}",
        candidate,
    )

    assert rendered == ("2. Dataset 3 (IMF) https://data.imf.org/x [1995-2024] Description 3. #3")


def test_the_reason_is_available_to_the_item_template() -> None:
    rendered = render_item("{name} - {reason}", _candidate(), reason="covers the subject")

    assert rendered == "Dataset 1 - covers the subject"


def test_the_reason_is_empty_when_none_was_given() -> None:
    assert render_item("{name}|{reason}", _candidate()) == "Dataset 1|"


def test_an_unknown_placeholder_renders_empty_rather_than_failing() -> None:
    """Templates come from channel config, so a typo must not break a chat turn."""
    assert render_item("{name}|{nonexistent}", _candidate()) == "Dataset 1|"


def test_an_item_cannot_reach_the_wrappers_items_placeholder() -> None:
    """`{items}` belongs to the wrapper; an item claiming it would render its own list."""
    assert render_item("{name}|{items}", _candidate()) == "Dataset 1|"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ blocks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_a_block_wraps_the_items_in_rank_order() -> None:
    selected = [(_candidate(1, rank=1), ""), (_candidate(2, rank=2), "")]

    block = render_block(_templates(item="- {name}"), selected)

    assert block == "### Datasets\n\n- Dataset 1\n- Dataset 2"


def test_an_empty_selection_renders_nothing_at_all() -> None:
    """Not even the wrapper: a header with no rows under it is worse than silence."""
    assert render_block(_templates(item="- {name}"), []) is None


def test_a_wrapper_that_renders_blank_is_reported_as_nothing() -> None:
    templates = _templates(item="", wrapper="{items}")

    assert render_block(templates, [(_candidate(), "")]) is None


def test_trailing_whitespace_in_a_multiline_item_template_is_trimmed() -> None:
    """Config templates are written as YAML block scalars, which keep a trailing newline."""
    selected = [(_candidate(1), ""), (_candidate(2), "")]

    block = render_block(_templates(item="- {name}\n"), selected)

    assert block == "### Datasets\n\n- Dataset 1\n- Dataset 2"

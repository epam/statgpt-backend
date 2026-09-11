"""
The RAG eval replays questions as of a cutoff date. `target_current_date` must replace
"today" everywhere the FILE_RAG tool uses it: in the prefilter it builds, and in the
configuration it sends to a RAG that accepts the override (Generic RAG).
Generic RAG silently drops unknown configuration keys, so the payload shape is asserted exactly.
"""

import datetime

import pytest

from statgpt.app.chains.file_rags.dial_rag import DialRagAgentFactory
from statgpt.app.chains.file_rags.dial_rag.prefilter import (
    DialRagPrefilterBuilder,
    PreFilterBuilder,
)
from statgpt.app.chains.file_rags.file_rag_tool import _RAG_IMPLEMENTATIONS, FileRagArgs
from statgpt.app.chains.file_rags.generic_rag import GenericRagAgentFactory
from statgpt.app.schemas.file_rags.dial_rag import (
    DialRagMetadata,
    DialRagState,
    PreFilterResponse,
    RagFilterDial,
    RagFilterDialSingle,
    TimePeriodFilter,
    TimePeriodFilterDial,
    TopNDocuments,
)
from statgpt.common.schemas import FileRagTool as FileRagToolConfig
from statgpt.common.schemas import RAGVersion
from statgpt.common.schemas.tool_details import FileRagDetails

CUTOFF = datetime.date(2025, 3, 15)


def _factory(version: RAGVersion) -> DialRagAgentFactory:
    tool_config = FileRagToolConfig(
        name="File_RAG", description="publications", details=FileRagDetails(version=version)
    )
    return _RAG_IMPLEMENTATIONS[version](tool_config, channel_config=None)  # type: ignore[arg-type]


@pytest.fixture
def prefilter() -> PreFilterResponse:
    return PreFilterResponse(
        rag_filter=RagFilterDial(
            filters=[
                RagFilterDialSingle(
                    publication_date=TimePeriodFilterDial(
                        start=datetime.date(2024, 1, 1), end=CUTOFF
                    ),
                    publication_type=None,
                ),
                RagFilterDialSingle(publication_date=None, publication_type="report"),
            ],
            top_n=TopNDocuments(limit=3),
        )
    )


EXPECTED_FILTERS = [
    {"publication_date": {"start": "2024-01-01", "end": "2025-03-15"}},
    {"publication_type": "report"},
]
EXPECTED_TOP_N = {"sort_by": ["publication_date"], "order": "desc", "limit": 3}


def test_registry_covers_every_rag_version():
    assert _RAG_IMPLEMENTATIONS[RAGVersion.DIAL] is DialRagAgentFactory
    assert _RAG_IMPLEMENTATIONS[RAGVersion.GENERIC] is GenericRagAgentFactory
    assert set(_RAG_IMPLEMENTATIONS) == set(RAGVersion)


def test_dial_rag_sends_flat_prefilter_and_ignores_current_date(prefilter):
    factory = _factory(RAGVersion.DIAL)
    expected = {
        "custom_fields": {"configuration": {"filters": EXPECTED_FILTERS, "top_n": EXPECTED_TOP_N}}
    }

    assert factory._build_extra_body(prefilter, None) == expected
    assert factory._build_extra_body(prefilter, CUTOFF) == expected
    assert factory._build_extra_body(PreFilterResponse(), CUTOFF) is None


def test_generic_rag_nests_prefilter_and_forwards_current_date(prefilter):
    factory = _factory(RAGVersion.GENERIC)
    selector = {"type": "explicit", "filters": EXPECTED_FILTERS, "top_n": EXPECTED_TOP_N}

    assert factory._build_extra_body(prefilter, None) == {
        "custom_fields": {"configuration": {"retriever": {"document_selector": selector}}}
    }
    assert factory._build_extra_body(prefilter, CUTOFF) == {
        "custom_fields": {
            "configuration": {
                "retriever": {"document_selector": selector},
                "generation": {"current_date": "2025-03-15"},
            }
        }
    }


def test_generic_rag_sends_current_date_without_prefilter():
    factory = _factory(RAGVersion.GENERIC)

    assert factory._build_extra_body(PreFilterResponse(), CUTOFF) == {
        "custom_fields": {"configuration": {"generation": {"current_date": "2025-03-15"}}}
    }
    assert factory._build_extra_body(PreFilterResponse(rag_filter=RagFilterDial()), CUTOFF) == {
        "custom_fields": {"configuration": {"generation": {"current_date": "2025-03-15"}}}
    }
    assert factory._build_extra_body(PreFilterResponse(), None) is None


def test_tool_state_carries_current_date_and_stays_backward_compatible():
    inputs = {"pre_filter": None, "metadata": None, "current_date": CUTOFF, "unrelated": object()}

    state = DialRagState(**{**inputs, "version": RAGVersion.GENERIC})

    assert state.version == RAGVersion.GENERIC
    assert state.current_date == CUTOFF
    assert DialRagState().current_date is None


def test_current_date_is_injected_not_exposed_to_llm():
    props = FileRagArgs.get_public_schema()["properties"]

    assert "query" in props
    assert "target_current_date" not in props
    assert "target_prefilter_json" not in props


# --- the prefilter statgpt builds itself must use the same "today"


def test_prompt_date_partials_use_reference_date():
    assert PreFilterBuilder._prompt_date_partials(CUTOFF) == {
        "current_date_long": "15 March 2025",
        "current_date_yyyymmdd": "2025-03-15",
        "current_year": "2025",
    }
    assert PreFilterBuilder._prompt_date_partials(None)["current_date_yyyymmdd"] == (
        datetime.date.today().isoformat()
    )


def test_fix_llm_hallucinations_respects_reference_date():
    # end date after the (mocked) today is dropped; against the wall clock it would be kept
    period = TimePeriodFilter(start="2024-01-01", end="2025-06-30")
    assert period.fix_llm_hallucinations(today=CUTOFF) == TimePeriodFilter(
        start="2024-01-01", end=""
    )
    assert period.fix_llm_hallucinations() == period

    # start date after the mocked today invalidates the whole filter
    future = TimePeriodFilter(start="2025-04-01", end="2025-06-30")
    assert future.fix_llm_hallucinations(today=CUTOFF).is_empty()


def test_latest_is_decoded_relative_to_reference_date():
    builder = DialRagPrefilterBuilder(
        metadata=DialRagMetadata(publication_types={"report"}),
        pub_type_to_decoder_mapping={"report": "-1y"},
    )

    rag_filter = builder.create_prefilter(
        publication_types=["report"],
        start_date=None,
        end_date=None,
        is_latest=True,
        last_n_publications=None,
        reference_date=CUTOFF,
    )

    assert rag_filter is not None
    (single,) = rag_filter.filters
    assert single.publication_type == "report"
    assert single.publication_date is not None
    assert single.publication_date.start == datetime.date(2024, 3, 15)
    assert single.publication_date.end is None

import datetime

from statgpt.app.schemas.file_rags.dial_rag import (
    DialRagMetadata,
    DialRagMetadataResponse,
    RagFilterDial,
    RagFilterDialSingle,
    SortOrder,
    TimePeriodFilterDial,
    TopNDocuments,
)
from statgpt.app.schemas.file_rags.generic_rag import GenericRagConfiguration


class TestDialRagMetadataResponseDimensions:
    """`dimensions` must accept both the DIAL RAG list shape and the Generic RAG map shape."""

    def test_list_shape(self):
        resp = DialRagMetadataResponse.model_validate(
            {
                "schema": {},
                "dimensions": [
                    {"name": "publication_type", "values": ["sigma", "SONAR"]},
                    {"name": "publication_date", "values": ["2024-01-01"]},
                ],
            }
        )
        meta = DialRagMetadata.from_response(resp)
        assert meta.publication_types == {"sigma", "SONAR"}
        assert meta.publication_dates == {"2024-01-01"}

    def test_map_shape(self):
        resp = DialRagMetadataResponse.model_validate(
            {
                "schema": {},
                "dimensions": {
                    "publication_type": ["sigma", "SONAR"],
                    "publication_date": ["2024-01-01"],
                },
            }
        )
        meta = DialRagMetadata.from_response(resp)
        assert meta.publication_types == {"sigma", "SONAR"}
        assert meta.publication_dates == {"2024-01-01"}


class TestGenericRagConfigurationFromRagFilterDial:
    def test_nests_filters_under_document_selector(self):
        rag_filter = RagFilterDial(
            filters=[
                RagFilterDialSingle(
                    publication_type="sigma",
                    publication_date=TimePeriodFilterDial(
                        start=datetime.date(2024, 1, 1), end=datetime.date(2024, 12, 31)
                    ),
                )
            ],
            top_n=TopNDocuments(sort_by=["publication_date"], order=SortOrder.desc, limit=10),
        )

        dumped = GenericRagConfiguration.from_rag_filter_dial(rag_filter).model_dump(
            mode="json", exclude_none=True
        )

        assert dumped == {
            "retriever": {
                "document_selector": {
                    "type": "explicit",
                    "filters": [
                        {
                            "publication_type": "sigma",
                            "publication_date": {"start": "2024-01-01", "end": "2024-12-31"},
                        }
                    ],
                    "top_n": {
                        "sort_by": ["publication_date"],
                        "order": "desc",
                        "limit": 10,
                    },
                }
            }
        }

    def test_omits_unset_fields(self):
        """A type-only filter with no top_n must not emit null date/top_n keys."""
        rag_filter = RagFilterDial(
            filters=[RagFilterDialSingle(publication_type="SONAR", publication_date=None)]
        )

        dumped = GenericRagConfiguration.from_rag_filter_dial(rag_filter).model_dump(
            mode="json", exclude_none=True
        )

        assert dumped == {
            "retriever": {
                "document_selector": {
                    "type": "explicit",
                    "filters": [{"publication_type": "SONAR"}],
                }
            }
        }

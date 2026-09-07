import datetime

from statgpt.app.chains.file_rags.dial_rag import DialRagAgentFactory
from statgpt.app.schemas import DialRagArtifact, DialRagState
from statgpt.app.schemas.file_rags.dial_rag import PreFilterResponse
from statgpt.app.schemas.file_rags.generic_rag import (
    GenericGenerationConfig,
    GenericRagConfiguration,
)
from statgpt.common.schemas import RAGVersion


class GenericRagAgentFactory(DialRagAgentFactory):
    """RAG factory for the Generic RAG DIAL application.

    Reuses the entire DIAL RAG flow (client, prefilter, metadata loading, streaming,
    attachment formatting). Only the RAG configuration payload differs: the Generic RAG
    application expects the prefilter nested under `retriever.document_selector`, and
    accepts a current date override for answer generation under `generation.current_date`.
    """

    def _build_extra_body(
        self, pre_filter_response: PreFilterResponse, current_date: datetime.date | None
    ) -> dict | None:
        rag_filter = pre_filter_response.rag_filter
        if rag_filter is not None and (rag_filter.filters or rag_filter.top_n is not None):
            config = GenericRagConfiguration.from_rag_filter_dial(rag_filter)
        else:
            config = GenericRagConfiguration()
        if current_date is not None:
            config.generation = GenericGenerationConfig(current_date=current_date)
        if config.is_empty():
            return None
        return {
            "custom_fields": {"configuration": config.model_dump(mode="json", exclude_none=True)}
        }

    @classmethod
    def _set_tool_state(cls, inputs: dict) -> dict:
        agent_state = DialRagState(**{**inputs, "version": RAGVersion.GENERIC})
        inputs[cls.FIELD_ARTIFACT] = DialRagArtifact(state=agent_state)
        return inputs

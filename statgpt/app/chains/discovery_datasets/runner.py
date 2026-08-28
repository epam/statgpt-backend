"""The chat-time lookup over a channel's published discovery datasets.

Runs concurrently with the data query pipeline and contributes a block of relevant discovery
datasets to its response. Deliberately not a `StatGptTool` and not a LangChain chain: nothing
calls it but the data query tool, and it has one entry point, so a plain class is both what the
call site needs and the smallest thing to wrap in a tool later.

It is decoration on someone else's answer, so it never fails the turn. Every failure is recorded
on the eval attachment and in the debug stage, and produces no block.
"""

import logging

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import StateVarsConfig
from statgpt.app.default_prompts import discovery_datasets_default_prompts
from statgpt.app.schemas.discovery_datasets import (
    DiscoveryCandidate,
    DiscoveryDatasetsEvalAttachment,
    DiscoveryDatasetsOutcome,
    DiscoveryRelevanceResponse,
)
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import ChannelConfig, DiscoveryDocumentMetadata, GenericRagDocument
from statgpt.common.schemas.discovery_datasets_tool import DiscoveryDatasetsDetails
from statgpt.common.services.generic_rag import GenericRagSearchClient
from statgpt.common.utils import write_yaml_to_stream
from statgpt.common.utils.async_utils import gather_with_concurrency
from statgpt.common.utils.models import get_chat_model

from .templates import render_block

_log = logging.getLogger(__name__)

_DOWNLOAD_CONCURRENCY = 8
"""How many document bodies to fetch at once.

They are short plain-text descriptions against one service, so this is about not opening a
connection per candidate rather than about protecting the service.
"""


class DiscoveryDatasetsRunner:
    """Retrieve, judge and render a channel's discovery datasets for one query."""

    def __init__(self, config: DiscoveryDatasetsDetails) -> None:
        self._config = config

    @classmethod
    def from_channel_config(cls, channel_config: ChannelConfig) -> "DiscoveryDatasetsRunner | None":
        """A runner for this channel, or `None` when the lookup is not configured for it."""
        if not channel_config.is_discovery_lookup_available:
            return None
        assert channel_config.discovery_datasets is not None  # implied by the check above
        return cls(channel_config.discovery_datasets.details)

    async def run(self, query: str, inputs: dict) -> DiscoveryDatasetsOutcome:
        """Look up discovery datasets relevant to `query`.

        Never raises: a failure is reported through the outcome, whose `rendered` is then `None`.
        """
        attachment = DiscoveryDatasetsEvalAttachment(query=query)
        try:
            rendered = await self._run(query, inputs, attachment)
        except Exception as e:
            _log.exception("Discovery datasets lookup failed")
            attachment.error = f"{type(e).__name__}: {e}"
            rendered = None

        attachment.rendered = rendered
        self._write_debug_stage(inputs, attachment)
        return DiscoveryDatasetsOutcome(rendered=rendered, eval_attachment=attachment)

    async def _run(
        self, query: str, inputs: dict, attachment: DiscoveryDatasetsEvalAttachment
    ) -> str | None:
        auth_context = ChainParameters.get_auth_context(inputs)

        candidates = await self._retrieve(query, auth_context)
        attachment.candidates = candidates
        if not candidates:
            return None

        response = await self._judge(query, candidates, auth_context)
        attachment.llm_response = response

        selected = self._select(candidates, response)
        attachment.selected_document_ids = [candidate.document_id for candidate, _ in selected]
        return render_block(self._config.templates, selected)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ retrieval ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _retrieve(self, query: str, auth_context: AuthContext) -> list[DiscoveryCandidate]:
        """Rank the channel's documents, then fill in the descriptions the ranking omits."""
        async with GenericRagSearchClient.for_application(
            self._config.get_application_id(), auth_context
        ) as client:
            documents = await client.search_documents(
                query, limit=self._config.top_n, indexes=self._config.indexes
            )
            if not documents:
                return []

            descriptions = await gather_with_concurrency(
                _DOWNLOAD_CONCURRENCY,
                *(self._download(client, document.id) for document in documents),
            )

        candidates = []
        for rank, (document, description) in enumerate(zip(documents, descriptions), start=1):
            candidate = self._to_candidate(document, rank, description)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    @staticmethod
    async def _download(client: GenericRagSearchClient, document_id: int) -> str:
        """The document's body, or an empty string if it could not be read.

        A candidate with no description is still worth judging on its metadata, which is where
        most of a discovery record lives, so one unreadable body does not cost the whole lookup.
        """
        try:
            return await client.download_document(document_id)
        except Exception:
            _log.exception(f"Failed to download discovery document {document_id}")
            return ""

    @staticmethod
    def _to_candidate(
        document: GenericRagDocument, rank: int, description: str
    ) -> DiscoveryCandidate | None:
        """One search hit as a candidate, or `None` if its metadata is not a discovery record.

        The RAG channel can hold documents this channel did not publish - both discovery grades
        and several StatGPT channels may share one - so a hit whose metadata does not fit is
        skipped rather than treated as a malformed discovery dataset.
        """
        try:
            metadata = DiscoveryDocumentMetadata.model_validate(document.metadata)
        except ValueError as e:
            _log.warning(
                f"Skipping discovery search hit {document.id}"
                f" ({document.display_name!r}): unexpected metadata: {e}"
            )
            return None

        return DiscoveryCandidate(
            document_id=document.id,
            rank=rank,
            display_name=document.display_name,
            metadata=metadata,
            description=description,
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ judging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _judge(
        self, query: str, candidates: list[DiscoveryCandidate], auth_context: AuthContext
    ) -> DiscoveryRelevanceResponse:
        prompt = (
            self._config.prompts.relevance_prompt
            or discovery_datasets_default_prompts.relevance_prompt
        )
        llm = get_chat_model(
            api_key=auth_context.api_key, model_config=self._config.llm_model_config
        ).with_structured_output(DiscoveryRelevanceResponse, method="json_schema")

        chain = prompt.get_template() | llm
        response = await chain.ainvoke(
            {"query": query, "candidates": self._format_candidates(candidates)}
        )
        assert isinstance(response, DiscoveryRelevanceResponse)  # narrows the structured output
        return response

    @staticmethod
    def _format_candidates(candidates: list[DiscoveryCandidate]) -> str:
        return write_yaml_to_stream([candidate.to_llm_dict() for candidate in candidates])

    @staticmethod
    def _select(
        candidates: list[DiscoveryCandidate], response: DiscoveryRelevanceResponse
    ) -> list[tuple[DiscoveryCandidate, str]]:
        """The candidates the model kept, in rank order.

        Verdicts on ids that were never offered are dropped: the model is asked to echo
        `document_id`s back, and one it invented must not be able to put a dataset in front of a
        user. Rank order rather than the model's order, so the list a user reads is the search's
        own ordering.
        """
        by_id = {candidate.document_id: candidate for candidate in candidates}
        reasons: dict[int, str] = {}
        for item in response.items:
            if not item.relevant:
                continue
            if item.document_id not in by_id:
                _log.warning(
                    f"!HALLUCINATION in discovery relevance judge! "
                    f"unexpected document_id: {item.document_id}"
                )
                continue
            reasons[item.document_id] = item.reason

        return [
            (candidate, reasons[candidate.document_id])
            for candidate in candidates
            if candidate.document_id in reasons
        ]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ debug stage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _write_debug_stage(self, inputs: dict, attachment: DiscoveryDatasetsEvalAttachment) -> None:
        """Report the whole run in one stage, opened only after the work is done.

        Opened at the end rather than around the work because this runs concurrently with the
        data query pipeline: a stage held open across the lookup would interleave with the
        pipeline's own stages in the user's view.
        """
        state = ChainParameters.get_state(inputs)
        if not state.get(StateVarsConfig.SHOW_DEBUG_STAGES):
            return

        choice = ChainParameters.get_choice(inputs)
        if choice is None:
            return

        content = attachment.model_dump_json(indent=2)
        try:
            with choice.create_stage(name=f"[DEBUG] {self._config.debug_stage_name}") as stage:
                stage.append_content(f"```json\n{content}\n```")
        except Exception:
            _log.exception("Failed to write the discovery datasets debug stage")

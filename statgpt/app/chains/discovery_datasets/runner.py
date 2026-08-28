"""The chat-time lookup over a channel's published discovery datasets.

Runs concurrently with the data query pipeline and contributes a block of relevant discovery
datasets to its response. Deliberately not a `StatGptTool` and not a LangChain chain: nothing
calls it but the data query tool, and it has one entry point, so a plain class is both what the
call site needs and the smallest thing to wrap in a tool later.

It is decoration on someone else's answer, so it never fails the turn. Every failure is recorded
on the eval attachment and in the debug stage, and produces no block.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Self

from pydantic import BaseModel

from statgpt.app.chains.parameters import ChainParameters
from statgpt.app.config import StateVarsConfig
from statgpt.app.default_prompts import discovery_datasets_default_prompts
from statgpt.app.schemas.discovery_datasets import (
    DiscoveryCandidate,
    DiscoveryDatasetsEvalAttachment,
    DiscoveryDatasetsOutcome,
    DiscoveryRelevanceResponse,
    SelectedDiscoveryDataset,
)
from statgpt.app.utils.dial_stages import DummyStage, StageI, delayed_timed_stage
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
    def from_channel_config(cls, channel_config: ChannelConfig) -> Self | None:
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
        with self._debug_stage(inputs) as stage:
            try:
                rendered = await self._run(query, inputs, attachment, stage)
            except Exception as e:
                _log.exception("Discovery datasets lookup failed")
                error = f"{type(e).__name__}: {e}"
                attachment.error = error
                self._report(stage, "## Error", error)
                rendered = None

        attachment.rendered = rendered
        return DiscoveryDatasetsOutcome(rendered=rendered, eval_attachment=attachment)

    async def _run(
        self,
        query: str,
        inputs: dict,
        attachment: DiscoveryDatasetsEvalAttachment,
        stage: StageI,
    ) -> str | None:
        auth_context = ChainParameters.get_auth_context(inputs)
        channel = ChainParameters.get_data_service(inputs).deployment_id

        candidates = await self._retrieve(query, auth_context, channel)
        attachment.candidates = candidates
        self._report(stage, "## Retrieved documents", self._as_yaml(candidates))
        if not candidates:
            return None

        response = await self._judge(query, candidates, auth_context)
        attachment.llm_response = response
        self._report(stage, "## Relevance verdicts", self._as_yaml(response.items))

        selected = self._select(candidates, response)
        attachment.selected_document_ids = [item.candidate.document_id for item in selected]
        rendered = render_block(self._config.templates, selected)
        self._report(stage, "## Rendered block", rendered or "_nothing was rendered_")
        return rendered

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ retrieval ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _retrieve(
        self, query: str, auth_context: AuthContext, channel: str
    ) -> list[DiscoveryCandidate]:
        """Rank the channel's documents, then fill in the descriptions the ranking omits."""
        async with GenericRagSearchClient.for_application(
            self._config.get_application_id(), auth_context
        ) as client:
            documents = await client.search_documents(
                query, limit=self._config.top_n, indexes=self._config.indexes
            )
            records = self._own_records(documents, channel)
            if not records:
                return []

            descriptions = await gather_with_concurrency(
                _DOWNLOAD_CONCURRENCY,
                *(self._download(client, document.id) for document, _ in records),
            )

        return [
            DiscoveryCandidate(
                document_id=document.id,
                rank=rank,
                display_name=document.display_name,
                metadata=metadata,
                description=description,
            )
            for rank, ((document, metadata), description) in enumerate(
                zip(records, descriptions), start=1
            )
        ]

    @staticmethod
    def _own_records(
        documents: list[GenericRagDocument], channel: str
    ) -> list[tuple[GenericRagDocument, DiscoveryDocumentMetadata]]:
        """The hits this channel published, each paired with its parsed metadata.

        One RAG channel is shared - by both discovery grades and by several StatGPT channels -
        so a search returns documents this channel did not publish. Two kinds are dropped here:
        a hit whose metadata is not a discovery record at all, and a hit that is one but belongs
        to another StatGPT channel. `statgpt_channel` is what the indexing job scopes publishing
        and withdrawal by, so it is the same identity on both sides.

        Filtered before the bodies are downloaded, so a foreign document costs no round trip.
        """
        records = []
        for document in documents:
            try:
                metadata = DiscoveryDocumentMetadata.model_validate(document.metadata)
            except ValueError as e:
                _log.warning(
                    f"Skipping discovery search hit {document.id}"
                    f" ({document.display_name!r}): unexpected metadata: {e}"
                )
                continue

            if metadata.statgpt_channel != channel:
                _log.debug(
                    f"Skipping discovery search hit {document.id}: published by channel"
                    f" {metadata.statgpt_channel!r}, not {channel!r}"
                )
                continue

            records.append((document, metadata))
        return records

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

    @classmethod
    def _format_candidates(cls, candidates: list[DiscoveryCandidate]) -> str:
        return cls._as_yaml([candidate.for_llm() for candidate in candidates])

    @staticmethod
    def _select(
        candidates: list[DiscoveryCandidate], response: DiscoveryRelevanceResponse
    ) -> list[SelectedDiscoveryDataset]:
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
            SelectedDiscoveryDataset(candidate=candidate, reason=reasons[candidate.document_id])
            for candidate in candidates
            if candidate.document_id in reasons
        ]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ debug stage ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @contextmanager
    def _debug_stage(self, inputs: dict) -> Iterator[StageI]:
        """A stage of this lookup's own, as if it were a tool the agent had called.

        Held open across the run so each step can report as soon as it has something to say.
        Safe to run beside the data query pipeline: a stage is its own slot in the choice, so
        only content written to the *same* stage can interleave. Delayed, so a lookup that
        reports nothing never opens an empty stage.
        """
        state = ChainParameters.get_state(inputs)
        choice = ChainParameters.get_choice(inputs)
        if not state.get(StateVarsConfig.SHOW_DEBUG_STAGES) or choice is None:
            yield DummyStage()
            return

        with delayed_timed_stage(choice, name=f"[DEBUG] {self._config.debug_stage_name}") as stage:
            yield stage

    @staticmethod
    def _report(stage: StageI, heading: str, content: str) -> None:
        """Add one section to the debug stage, without letting the stage cost the lookup."""
        try:
            stage.append_content(f"{heading}\n\n{content}\n\n")
        except Exception:
            _log.exception("Failed to write the discovery datasets debug stage")

    @staticmethod
    def _as_yaml(models: Sequence[BaseModel]) -> str:
        dumped = write_yaml_to_stream([model.model_dump(mode="json") for model in models])
        return f"```yaml\n{dumped}```"

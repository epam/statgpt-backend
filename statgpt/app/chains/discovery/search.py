"""Searching the discovery index and judging what is worth referring to.

Four steps, in one place so the fallback that calls it stays a few lines: ground the request's
countries against the values the channel holds, retrieve with a document filter built from them,
fold the retrieved chunks back into datasets, and let a judge decide which datasets to surface.

Written as a component rather than as logic inside the data query tool, so that the standalone
discovery search tool of the full design becomes a second caller rather than a rewrite.
"""

import logging
import re

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from openai import APIError

from statgpt.app.chains.discovery import prompts
from statgpt.app.schemas.discovery import (
    DiscoveryCandidate,
    DiscoveryDocumentSelector,
    DiscoveryFilterEntry,
    DiscoveryGenerationConfig,
    DiscoveryJudgement,
    DiscoveryReferralItem,
    DiscoveryRetrieverConfig,
    DiscoverySearchConfiguration,
    DiscoverySearchResult,
)
from statgpt.app.utils import openai as openai_utils
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import DiscoveryFallbackConfig
from statgpt.common.schemas.generic_rag import GenericRagDocument
from statgpt.common.services import GenericRagIngestionClient, ground_reference_areas
from statgpt.common.utils.models import get_chat_model

_log = logging.getLogger(__name__)

_AREA_FIELD = "reference_area_values"
"""The metadata field the country pre-filter matches on."""

_CITATION_PREFIX = re.compile(r"^\s*\[\d+\]\s*")
"""The '[3] ' the application prefixes to an attachment title.

`create_attachment` titles an attachment `f"[{citation_index}] {doc.source_display_name}"`, so
the display name - the only handle the response gives on which document a chunk came from - has
to be recovered by stripping it.
"""


class DiscoverySearchService:
    """One discovery search, against one channel's discovery RAG application.

    Holds no state between searches beyond its configuration, but does own an HTTP client for
    the channel API, so use it as an async context manager.
    """

    def __init__(
        self,
        *,
        config: DiscoveryFallbackConfig,
        application_id: str,
        statgpt_channel: str,
        auth_context: AuthContext,
    ) -> None:
        self._config = config
        self._application_id = application_id
        self._statgpt_channel = statgpt_channel
        self._auth_context = auth_context
        self._client = GenericRagIngestionClient.for_application(application_id)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "DiscoverySearchService":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ entry point ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def search(self, question: str, countries: list[str]) -> DiscoverySearchResult:
        """Find the datasets worth referring `question` to.

        `countries` are the surface forms a request named - the entities the data query pipeline
        already extracted before it failed - not codes and not grounded values.
        """
        documents = await self._client.list_documents()
        mine = {
            document.display_name: document
            for document in documents
            if self._is_ours(document) and document.display_name
        }

        grounded = ground_reference_areas(countries, self._available_areas(mine.values()))
        if grounded.unmatched:
            _log.info(
                f"Discovery search: no indexed reference area matches {grounded.unmatched};"
                f" searching {'within ' + str(grounded.values) if grounded.values else 'unfiltered'}"
            )

        chunks_by_document = await self._retrieve(question, grounded.values)
        candidates = self._fold(chunks_by_document, mine)

        result = DiscoverySearchResult(
            grounded_areas=grounded.values,
            unmatched_areas=grounded.unmatched,
            retrieved=len(candidates),
        )
        if not candidates:
            return result

        result.items = await self._judge(question, candidates)
        return result

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ filtering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _is_ours(self, document: GenericRagDocument) -> bool:
        """Whether this document belongs to this channel and the grade being searched.

        One RAG application serves several StatGPT channels and both grades, so without this a
        search would refer the user to another channel's datasets.
        """
        metadata = document.metadata
        return (
            metadata.get("grade") == self._config.grade
            and metadata.get("statgpt_channel") == self._statgpt_channel
        )

    @staticmethod
    def _available_areas(documents) -> list[str]:
        """The country values this channel's own documents carry.

        Read from the documents rather than from `/channel/metadata`, whose dimensions cover the
        whole application - every StatGPT channel and both grades. Filtering on a value that
        exists only in another channel's records would narrow this search to nothing.

        This also satisfies the service's hard requirement on filter values: it types a
        filterable field as a `Literal` of the values present, so a value no document carries
        fails the whole retrieval request rather than matching nothing.
        """
        values: list[str] = []
        for document in documents:
            entries = document.metadata.get(_AREA_FIELD)
            if isinstance(entries, list):
                values.extend(entry for entry in entries if isinstance(entry, str) and entry)
        return list(dict.fromkeys(values))

    def _build_configuration(self, areas: list[str]) -> DiscoverySearchConfiguration:
        """Build the retrieval request's document filter.

        One entry per country, OR'd by the service, each carrying the grade and channel that the
        service AND's within the entry. No grounded country means one entry with the country axis
        left out: an unfiltered search is a precision problem the judge absorbs, while an
        over-narrow filter loses records irrecoverably.
        """
        base = {"grade": self._config.grade, "statgpt_channel": self._statgpt_channel}
        if areas:
            entries = [DiscoveryFilterEntry(**base, reference_area_values=area) for area in areas]
        else:
            entries = [DiscoveryFilterEntry(**base)]

        return DiscoverySearchConfiguration(
            retriever=DiscoveryRetrieverConfig(
                document_selector=DiscoveryDocumentSelector(filters=entries)
            ),
            generation=DiscoveryGenerationConfig(),
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ retrieval ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _retrieve(self, question: str, areas: list[str]) -> dict[str, list[str]]:
        """Retrieve chunks, grouped by the display name of the document they came from.

        The application is asked for `retrieval_only` generation, so it answers with the
        retrieval results as attachments and never calls an LLM. Each attachment carries the
        chunk text and a title naming its document; the document metadata is not in the response,
        which is why the caller pairs these names with a document listing.
        """
        configuration = self._build_configuration(areas)
        client = openai_utils.get_async_client(api_key=self._auth_context.api_key)

        try:
            response = await client.chat.completions.create(
                model=self._application_id,
                messages=[{"role": "user", "content": question}],
                extra_body=configuration.as_extra_body(),
            )
        except APIError:
            _log.exception("Discovery retrieval failed")
            return {}

        grouped: dict[str, list[str]] = {}
        for attachment in self._response_attachments(response):
            title = attachment.get("title")
            if not isinstance(title, str):
                continue
            name = _CITATION_PREFIX.sub("", title).strip()
            data = attachment.get("data")
            if name:
                grouped.setdefault(name, []).append(data if isinstance(data, str) else "")
        return grouped

    @staticmethod
    def _response_attachments(response) -> list[dict]:
        """Pull the attachments out of a non-streaming DIAL chat completion.

        `custom_content` is DIAL's extension to the OpenAI schema, so the client parses it into
        `model_extra` rather than into a declared field.
        """
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError):
            return []

        extra = getattr(message, "model_extra", None) or {}
        custom_content = extra.get("custom_content")
        if custom_content is None:
            custom_content = getattr(message, "custom_content", None)
        if not isinstance(custom_content, dict):
            return []

        attachments = custom_content.get("attachments")
        return [item for item in attachments if isinstance(item, dict)] if attachments else []

    def _fold(
        self, chunks_by_document: dict[str, list[str]], mine: dict[str, GenericRagDocument]
    ) -> list[DiscoveryCandidate]:
        """Turn retrieved chunks into one candidate per dataset.

        Retrieval returns chunks and a referral is about datasets, so chunks are grouped by their
        document and the candidate is built from that document's full metadata - the judge and the
        referral both need fields the matching chunk may not contain.

        A name that matches no document of ours is skipped: the filter should have excluded it,
        and a candidate whose metadata cannot be read is one the referral could not link to.
        """
        candidates: list[DiscoveryCandidate] = []
        for name, chunks in chunks_by_document.items():
            document = mine.get(name)
            if document is None:
                _log.warning(
                    f"Discovery search retrieved document {name!r},"
                    f" which is not an indexed record of this channel and grade"
                )
                continue
            candidates.append(DiscoveryCandidate.from_document(document, chunks))
        return candidates[: self._config.max_candidates]

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ judging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _judge(
        self, question: str, candidates: list[DiscoveryCandidate]
    ) -> list[DiscoveryReferralItem]:
        """Pick the datasets worth showing, or none.

        A judge is not optional here. Retrieval always returns something, so without this step
        every request that reaches the fallback would produce a referral - including the many
        whose subject nothing in the index covers.
        """
        try:
            judgement: DiscoveryJudgement = await self._judge_chain().ainvoke(
                {
                    "question": question,
                    "candidates": self._render_candidates(candidates),
                    "max_referrals": self._config.max_referrals,
                }
            )
        except Exception:
            # A referral is an extra on top of a no-data answer. Failing to produce one must not
            # turn the tool's response into an error.
            _log.exception("Discovery relevance judge failed; referring to nothing")
            return []

        items: list[DiscoveryReferralItem] = []
        for selection in judgement.selections:
            index = selection.index - 1
            if not 0 <= index < len(candidates):
                _log.warning(
                    f"Discovery judge selected candidate {selection.index},"
                    f" which is outside the {len(candidates)} it was given"
                )
                continue
            items.append(
                DiscoveryReferralItem(
                    candidate=candidates[index],
                    reason=selection.reason,
                    missing=selection.missing,
                )
            )
            if len(items) >= self._config.max_referrals:
                break
        return items

    def _judge_chain(self) -> Runnable:
        llm = get_chat_model(
            api_key=self._auth_context.api_key,
            model_config=self._config.judge_model_config,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(prompts.JUDGE_SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(prompts.JUDGE_USER_PROMPT),
            ]
        )
        return prompt | llm.with_structured_output(schema=DiscoveryJudgement, method="json_schema")

    @staticmethod
    def _render_candidates(candidates: list[DiscoveryCandidate]) -> str:
        """Render the candidate list the judge reads.

        Every workbook field the judge needs is labeled, and the two negative ones are labeled as
        exclusions rather than by their workbook names, so their meaning does not depend on the
        judge inferring it from a column heading.
        """
        blocks: list[str] = []
        for number, candidate in enumerate(candidates, start=1):
            lines = [f"{number}. {candidate.label}"]
            fields = (
                ("Agency", candidate.agency),
                ("Countries / areas", candidate.reference_area),
                ("Sub-national breakdown", candidate.regional_coverage),
                ("Excluded regions", candidate.excluded_regional_values),
                ("Time coverage", candidate.time_coverage),
                ("Frequencies", candidate.frequency_coverage),
                ("Indicators", candidate.indicators_coverage),
                ("Indicators NOT present", candidate.missing_indicators),
            )
            lines.extend(f"   {label}: {value}" for label, value in fields if value)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

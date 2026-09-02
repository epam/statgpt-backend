"""Narrowing the discovery search's candidate set before it is ranked.

A channel publishes datasets from many agencies, about many countries, at many frequencies, and
the search's `top_n` is spent on whatever ranks highest across all of them. A query that names a
country is far better served by ranking that country's datasets alone - so the query is read for
values on several axes, and the search is asked to consider only documents carrying them.

Everything here can only remove candidates, and every step can fail on its own, so the whole
module is written to degrade instead of raising: an axis that produces nothing drops out, and a
lookup where nothing survives searches the channel exactly as it did before pre-filtering
existed. The caller gets a matcher or `None`, never an exception.

Two properties of the RAG channel's filter language shape the result:

A filter entry carries one value per field and entries are OR-ed, so several values on several
axes are a cross product of entries rather than one entry holding lists.

The values a request may carry are exactly the ones the channel currently holds - the service
types each filterable field as a `Literal` over its own dimensions. A value outside them fails
the whole search request, not just its own clause, which is why every selection is intersected
with `dimensions` before it is allowed into a matcher.

The two area axes are the one place where axes are not independent. A record covering the United
States as its subject and one covering it as a trade partner are both answers to a query naming
it, so the two are alternatives to each other: they contribute to the same slot of the cross
product rather than being multiplied together, which would demand both at once.
"""

import asyncio
import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from langchain_core.runnables import RunnableParallel, RunnableSerializable
from langchain_openai import AzureChatOpenAI

from statgpt.app.default_prompts import discovery_datasets_default_prompts
from statgpt.app.schemas.discovery_datasets import (
    DiscoveryAxisSelection,
    DiscoveryPreFilterAxisReport,
    DiscoveryPreFilterReport,
)
from statgpt.app.settings.dial_app import dial_app_settings
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas import (
    REFERENCE_AREA_KIND,
    DiscoveryPreFilterAxis,
    GenericRagDocumentFilter,
    GenericRagDocumentMatcher,
    ReferenceAreaRole,
    SystemUserPrompt,
)
from statgpt.common.schemas.discovery_datasets_tool import DiscoveryDatasetsDetails
from statgpt.common.services.generic_rag import GenericRagSearchClient
from statgpt.common.utils import FREQUENCY_VOCABULARY, AsyncLoadingCache
from statgpt.common.utils.models import get_chat_model

_log = logging.getLogger(__name__)

_AXIS_FIELDS: dict[DiscoveryPreFilterAxis, str] = {
    DiscoveryPreFilterAxis.REFERENCE_AREA: "parsed_reference_areas",
    DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA: "parsed_partner_reference_areas",
    DiscoveryPreFilterAxis.FREQUENCY: "parsed_frequencies",
    DiscoveryPreFilterAxis.AGENCY: "agency",
}
"""The discovery document field each axis's values are matched against."""

_AREA_AXIS_ROLES: dict[DiscoveryPreFilterAxis, ReferenceAreaRole] = {
    DiscoveryPreFilterAxis.REFERENCE_AREA: ReferenceAreaRole.SUBJECT,
    DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA: ReferenceAreaRole.PARTNER,
}
"""The axes resolved through the vocabulary channel, and the role each is offered.

Both read the same channel, filtered to their own role: a label no record names as a partner is
not a value the partner axis may narrow by, and offering it would spend the vocabulary search's
limit on candidates grounding would drop anyway.
"""

_AREA_GROUP = "reference_area"
"""The cross-product slot the two area axes share - see `_cross_product`."""

_UNCONFIGURED_VOCABULARY = "no reference-area vocabulary is configured for this channel"
_UNREADABLE_VOCABULARY = "the reference-area vocabulary could not be read"

_Assignment = tuple[str, str]
"""One filterable field and the single value an entry may carry on it."""

_dimensions_cache: AsyncLoadingCache[dict[str, list[str]]] = AsyncLoadingCache(
    ttl=dial_app_settings.discovery_datasets_pre_filter_dimensions_ttl
)
"""The filterable values of each discovery channel, keyed by application id.

Module-level because a runner is built per turn: an instance-level cache would load the
dimensions again for every query. Shared across users deliberately - this describes a channel's
documents, not anyone's access to them, and the channel endpoints authorize each request on its
own key regardless.

A cached list can fall behind the channel, which shows up as a request the service rejects. That
is a fallback, not a fault: the alternative is a metadata read on the critical path of every
turn.
"""


def forget_dimensions(application_id: str) -> None:
    """Drop a channel's cached filterable values, so the next lookup reads them again.

    Called when a narrowed search is rejected, which is what a stale entry looks like. Without
    this the entry would stand for the rest of its TTL, and every lookup in that window would
    pay a rejected search and a fallback rather than only the first one.
    """
    _dimensions_cache.remove(application_id)


@dataclass(frozen=True)
class _Vocabulary:
    """The values an axis may select from, and why there are none when there are none.

    The two reasons an area axis offers nothing are worth telling apart: one is a channel
    configured without a vocabulary, the other a vocabulary that could not be read. Neither
    stops the other axes, and both are worth reading in an eval attachment.
    """

    values: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass(frozen=True)
class DiscoveryPreFilter:
    """The narrowed search a lookup should attempt, and the account of how it was arrived at."""

    matcher: GenericRagDocumentMatcher | None
    """The matcher to search with, or `None` when nothing survived and the caller should not."""

    report: DiscoveryPreFilterReport
    """What each axis offered, chose and kept. Recorded on the eval attachment either way."""


class DiscoveryPreFilterBuilder:
    """Reads a query for the values its answer can be narrowed to, and builds the matcher.

    One instance per lookup. Never raises: see the module docstring.
    """

    def __init__(self, config: DiscoveryDatasetsDetails) -> None:
        self._config = config
        self._pre_filter = config.pre_filter

    async def build(
        self, query: str, auth_context: AuthContext, channel: str
    ) -> DiscoveryPreFilter:
        """The pre-filter for one query, or a report saying why there is none."""
        report = DiscoveryPreFilterReport(enabled=self._pre_filter.enabled)
        if not self._pre_filter.enabled:
            report.fallback_reason = "pre-filtering is disabled for this channel"
            return DiscoveryPreFilter(matcher=None, report=report)

        try:
            return await self._build(query, auth_context, channel, report)
        except Exception as e:
            _log.exception("Discovery pre-filter failed")
            report.fallback_reason = f"{type(e).__name__}: {e}"
            return DiscoveryPreFilter(matcher=None, report=report)

    async def _build(
        self,
        query: str,
        auth_context: AuthContext,
        channel: str,
        report: DiscoveryPreFilterReport,
    ) -> DiscoveryPreFilter:
        axes = self._axes()
        dimensions, areas = await self._vocabularies(query, auth_context, channel, axes)

        if dimensions is None:
            report.fallback_reason = (
                "the discovery channel's filterable values could not be read, so no selection"
                " could be grounded"
            )
            return DiscoveryPreFilter(matcher=None, report=report)

        offered = {axis: self._offered(axis, dimensions, areas) for axis in axes}
        selections = await self._select(query, auth_context, axes, offered)

        groups: dict[str, list[_Assignment]] = {}
        for axis in axes:
            axis_report = DiscoveryPreFilterAxisReport(
                axis=axis, offered=offered[axis], selected=selections.get(axis, [])
            )
            if (vocabulary := areas.get(axis)) is not None:
                axis_report.error = vocabulary.reason
            assignments = self._ground(axis, axis_report.selected, dimensions)
            axis_report.grounded = sorted({value for _, value in assignments})
            report.axes.append(axis_report)
            if assignments:
                groups.setdefault(_group_of(axis), []).extend(assignments)

        filters = self._cross_product(channel, list(groups.values()))
        report.filters = len(filters)
        if not filters:
            report.fallback_reason = (
                "the query named no value this channel holds on any of the enabled axes"
            )
            return DiscoveryPreFilter(matcher=None, report=report)

        return DiscoveryPreFilter(matcher=GenericRagDocumentMatcher(filters=filters), report=report)

    def _axes(self) -> list[DiscoveryPreFilterAxis]:
        """The configured axes, de-duplicated and in configuration order.

        An area axis stays in the list even when the channel publishes no vocabulary: it then
        reports why it contributed nothing, which is more use than silently disappearing.
        """
        seen: dict[DiscoveryPreFilterAxis, None] = {}
        for axis in self._pre_filter.axes:
            seen.setdefault(axis, None)
        return list(seen)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ vocabularies ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _vocabularies(
        self,
        query: str,
        auth_context: AuthContext,
        channel: str,
        axes: Sequence[DiscoveryPreFilterAxis],
    ) -> tuple[dict[str, list[str]] | None, dict[DiscoveryPreFilterAxis, _Vocabulary]]:
        """The channel's filterable values and each area axis's candidates.

        Both are network reads and neither depends on the other, so they run together. They fail
        differently: without `dimensions` nothing can be grounded and the whole pre-filter is
        off, while a failed vocabulary search costs only the axis it was for.
        """
        area_axes = [axis for axis in axes if axis in _AREA_AXIS_ROLES]
        read_dimensions, read_areas = await asyncio.gather(
            self._dimensions(auth_context),
            self._area_candidates(query, auth_context, channel, area_axes),
            return_exceptions=True,
        )

        dimensions: dict[str, list[str]] | None = None
        if isinstance(read_dimensions, BaseException):
            _log.warning(
                "Could not read the discovery channel's dimensions", exc_info=read_dimensions
            )
        else:
            dimensions = read_dimensions

        if isinstance(read_areas, BaseException):
            _log.warning("Could not read the reference-area vocabulary", exc_info=read_areas)
            areas = {axis: _Vocabulary(reason=_UNREADABLE_VOCABULARY) for axis in area_axes}
        else:
            areas = read_areas

        return dimensions, areas

    async def _dimensions(self, auth_context: AuthContext) -> dict[str, list[str]]:
        """The values the discovery channel currently holds, per filterable field. Cached."""
        application_id = self._config.get_application_id()

        async def load() -> dict[str, list[str]]:
            async with GenericRagSearchClient.for_application(
                application_id, auth_context
            ) as client:
                return (await client.get_metadata_schema()).dimensions

        return await _dimensions_cache.get(application_id, load)

    async def _area_candidates(
        self,
        query: str,
        auth_context: AuthContext,
        channel: str,
        area_axes: Sequence[DiscoveryPreFilterAxis],
    ) -> dict[DiscoveryPreFilterAxis, _Vocabulary]:
        """The vocabulary each area axis may select from.

        One search per axis, over the same channel and on the same client, concurrently: the two
        ask the same question of different halves of the vocabulary, and giving each its own
        limit is the point of tagging the labels with their roles in the first place.

        A search can fail on its own account - a channel whose records name no partner holds no
        `partner` among the vocabulary's role dimensions, so the service rejects that filter
        value outright. It costs its own axis and nothing else, which is the right outcome: an
        axis with no labels behind it had nothing to narrow by.
        """
        if not area_axes:
            return {}

        application_id = self._config.get_reference_area_application_id()
        if application_id is None:
            return {axis: _Vocabulary(reason=_UNCONFIGURED_VOCABULARY) for axis in area_axes}

        async with GenericRagSearchClient.for_application(application_id, auth_context) as client:
            found = await asyncio.gather(
                *(
                    self._role_candidates(client, query, channel, _AREA_AXIS_ROLES[axis])
                    for axis in area_axes
                ),
                return_exceptions=True,
            )

        vocabularies: dict[DiscoveryPreFilterAxis, _Vocabulary] = {}
        for axis, values in zip(area_axes, found):
            if isinstance(values, BaseException):
                _log.warning(f"Could not read the {axis} vocabulary", exc_info=values)
                vocabularies[axis] = _Vocabulary(reason=_UNREADABLE_VOCABULARY)
            else:
                vocabularies[axis] = _Vocabulary(values=values)
        return vocabularies

    async def _role_candidates(
        self, client: GenericRagSearchClient, query: str, channel: str, role: ReferenceAreaRole
    ) -> list[str]:
        """The labels of one role most like the query.

        Searched rather than listed: a channel holds hundreds of labels, and a model asked to
        pick from all of them at once is both slower and less accurate than one choosing from
        the handful the query actually resembles.

        The whole query is the search text, not a guess at the part naming a country: the
        documents are one line each, so a word naming no label matches nothing rather than
        adding noise.
        """
        matcher = GenericRagDocumentMatcher(
            filters=[
                GenericRagDocumentFilter(
                    kind=REFERENCE_AREA_KIND, statgpt_channel=channel, roles=role
                )
            ]
        )
        documents = await client.search_documents(
            query, limit=self._pre_filter.reference_area_top_n, matcher=matcher
        )

        values: list[str] = []
        for document in documents:
            value = document.metadata.get("value")
            if isinstance(value, str) and value.strip() and value not in values:
                values.append(value)
        return values

    @staticmethod
    def _offered(
        axis: DiscoveryPreFilterAxis,
        dimensions: dict[str, list[str]],
        areas: dict[DiscoveryPreFilterAxis, _Vocabulary],
    ) -> list[str]:
        """The values one axis may select from.

        Three sources, one per kind of axis: an area axis is offered what its own vocabulary
        search returned, frequency the template's closed list, and anything else the values the
        discovery channel itself holds on the field that axis filters.
        """
        if (vocabulary := areas.get(axis)) is not None:
            return vocabulary.values
        if axis is DiscoveryPreFilterAxis.FREQUENCY:
            return list(FREQUENCY_VOCABULARY)
        return dimensions.get(_AXIS_FIELDS[axis], [])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ selection ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    async def _select(
        self,
        query: str,
        auth_context: AuthContext,
        axes: Sequence[DiscoveryPreFilterAxis],
        offered: dict[DiscoveryPreFilterAxis, list[str]],
    ) -> dict[DiscoveryPreFilterAxis, list[str]]:
        """Ask the model, one sub-chain per axis, which of the offered values the query names.

        One chain per axis rather than one call answering all of them: each is a small, closed
        question, and a chain given only its own vocabulary cannot answer an axis with another
        axis's values. An axis with nothing to offer is not asked at all.
        """
        askable = [axis for axis in axes if offered.get(axis)]
        if not askable:
            return {}

        llm = get_chat_model(
            api_key=auth_context.api_key, model_config=self._pre_filter.llm_model_config
        )
        chains = {
            axis.value: self._chain(llm, self._prompt(axis), offered[axis]) for axis in askable
        }
        # Left to propagate: `RunnableParallel` fails as a whole anyway, so there is no partial
        # answer to salvage, and `build` turns it into a fallback whose reason names the failure
        # rather than reporting that the query happened to name nothing.
        answers = await RunnableParallel(chains).ainvoke({"query": query})
        return {axis: _values_of(answers.get(axis.value)) for axis in askable}

    @staticmethod
    def _chain(
        llm: AzureChatOpenAI, prompt: SystemUserPrompt, values: Sequence[str]
    ) -> RunnableSerializable:
        """One axis's sub-chain, its vocabulary baked in as a partial.

        Partialled rather than passed at invocation time because the branches of a
        `RunnableParallel` all receive the same input: the query is shared, the vocabulary is not.
        """
        template = prompt.get_template().partial(values="\n".join(f"- {value}" for value in values))
        return template | llm.with_structured_output(DiscoveryAxisSelection, method="json_schema")

    def _prompt(self, axis: DiscoveryPreFilterAxis) -> SystemUserPrompt:
        configured = self._pre_filter.prompts
        defaults = discovery_datasets_default_prompts
        match axis:
            case DiscoveryPreFilterAxis.REFERENCE_AREA:
                return configured.reference_area_prompt or defaults.reference_area_prompt
            case DiscoveryPreFilterAxis.PARTNER_REFERENCE_AREA:
                return (
                    configured.partner_reference_area_prompt
                    or defaults.partner_reference_area_prompt
                )
            case DiscoveryPreFilterAxis.FREQUENCY:
                return configured.frequency_prompt or defaults.frequency_prompt
            case DiscoveryPreFilterAxis.AGENCY:
                return configured.agency_prompt or defaults.agency_prompt

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ grounding ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def _ground(
        axis: DiscoveryPreFilterAxis,
        selected: Sequence[str],
        dimensions: dict[str, list[str]],
    ) -> list[_Assignment]:
        """The field assignments one axis's selection is allowed to produce.

        A value is kept only if the target field's dimensions hold it, and it is kept under the
        channel's own spelling: the service compares filter values exactly, so a re-cased answer
        would be rejected along with the whole request rather than simply not matching.

        Matching case-insensitively is what makes that recovery possible at all - the model is
        given the values verbatim and usually returns them verbatim, but a re-cased answer is a
        correct answer, and dropping it would narrow nothing where it should narrow well.
        """
        name = _AXIS_FIELDS[axis]
        known = {value.casefold(): value for value in dimensions.get(name, [])}
        # De-duplicated: two selections folding onto one channel value are one requirement, and
        # keeping both would multiply the cross product by entries identical to each other.
        grounded: dict[str, None] = {}
        for value in selected:
            canonical = known.get(value.casefold())
            if canonical is not None:
                grounded.setdefault(canonical, None)
        return [(name, value) for value in grounded]

    @staticmethod
    def _cross_product(
        channel: str, groups: Sequence[Sequence[_Assignment]]
    ) -> list[GenericRagDocumentFilter]:
        """Every combination of one assignment per surviving group, as one filter entry each.

        The entries are OR-ed by the service and the fields within an entry are AND-ed, so this
        is what "in one of these countries, at one of these frequencies" has to be expressed as.

        A group is one requirement the query makes, which is not always one axis: covering an
        area as a subject and covering it as a partner are two ways of satisfying "about this
        country", so the two area axes arrive here as one group and become alternative entries
        instead of a demand for both at once.

        The channel scope is repeated in every entry: without it, one entry would let another
        channel's documents through the whole matcher.
        """
        if not groups:
            return []
        return [
            GenericRagDocumentFilter(statgpt_channel=channel, **dict(combination))
            for combination in itertools.product(*groups)
        ]


def _group_of(axis: DiscoveryPreFilterAxis) -> str:
    """Which of the query's requirements an axis helps satisfy - see `_cross_product`."""
    return _AREA_GROUP if axis in _AREA_AXIS_ROLES else axis.value


def _values_of(answer: object) -> list[str]:
    """The values one sub-chain answered with, de-duplicated, or nothing if it answered oddly."""
    if not isinstance(answer, DiscoveryAxisSelection):
        return []
    # Stripped rather than merely tested, for the reason `_ground` matches case-insensitively:
    # a value the model padded with whitespace is a correct answer, and grounding it verbatim
    # would drop it against a channel that holds the same value unpadded.
    return list(dict.fromkeys(stripped for value in answer.values if (stripped := value.strip())))

from datetime import date
from typing import TypeVar

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableParallel, RunnableSerializable
from langchain_openai import AzureChatOpenAI

from common import utils
from common.config import multiline_logger as logger
from statgpt.default_prompts import DialRagPrompts
from statgpt.schemas.file_rags.dial_rag import (
    DialRagMetadata,
    LastNPublicationsFilter,
    LatestFilter,
    PreFilterResponse,
    PublicationTypesFilter,
    RagFilterDial,
    RagFilterDialSingle,
    RagFilterLLMOutput,
    SingleFilterLLMOutput,
    SortOrder,
    TimePeriodFilter,
    TimePeriodFilterDial,
    TopNDocuments,
)

FilterBuilderResultType = TypeVar("FilterBuilderResultType", bound=SingleFilterLLMOutput)


class SingleFilterChainBuilder:

    @staticmethod
    def create_chain(
        llm,
        system_prompt: str,
        output_type: type[FilterBuilderResultType],
        partials: dict | None = None,
    ) -> RunnableSerializable:
        """Returns the runnable chain for this filter builder."""
        if partials is None:
            partials = {}
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("{query}"),
            ]
        ).partial(**partials)
        chain = prompt | llm.with_structured_output(schema=output_type, method='json_schema')
        return chain


class PreFilterBuilder:

    def __init__(
        self,
        llm: AzureChatOpenAI,
        metadata: DialRagMetadata,
        pub_type_to_decoder_mapping: dict[str, str],
    ):
        self._llm = llm
        self._dial_rag_prefilter_builder = DialRagPrefilterBuilder(
            metadata=metadata,
            pub_type_to_decoder_mapping=pub_type_to_decoder_mapping,
        )
        self._metadata = metadata

    async def build_filter_from_query(self, query: str) -> PreFilterResponse:
        llm_filters = await self._call_llm(query=query)
        dial_rag_filters = self._convert_llm_output_to_dial_rag_filter(llm_filters)
        return PreFilterResponse(llm_output=llm_filters, rag_filter=dial_rag_filters)

    async def _call_llm(self, query: str) -> RagFilterLLMOutput:
        pre_filter_chain = self._create_prefilter_chain()
        outputs = await pre_filter_chain.ainvoke({"query": query})
        return RagFilterLLMOutput(
            time_period=outputs["time_period"],
            is_latest=outputs["latest"].is_latest,
            publication_types=outputs["publication_types"].publication_types,
            last_n_publications=outputs["last_n_publications"].last_n_publications,
        )

    def _create_prefilter_chain(self) -> RunnableSerializable:
        date_chain = SingleFilterChainBuilder.create_chain(
            llm=self._llm,
            system_prompt=DialRagPrompts.PREFILTER_SYSTEM_PROMPT_DATE,
            output_type=TimePeriodFilter,
            partials={
                "current_date_long": utils.get_today_date_long(),
                "current_date_yyyymmdd": utils.get_ts_now_str(ts_format="%Y-%m-%d"),
            },
        )
        latest_chain = SingleFilterChainBuilder.create_chain(
            llm=self._llm,
            system_prompt=DialRagPrompts.PREFILTER_SYSTEM_PROMPT_LATEST,
            output_type=LatestFilter,
        )
        publication_types_chain = SingleFilterChainBuilder.create_chain(
            llm=self._llm,
            system_prompt=DialRagPrompts.PREFILTER_SYSTEM_PROMPT_PUBLICATIONS,
            output_type=PublicationTypesFilter,
            partials={
                "available_publication_types": list(self._metadata.publication_types),
            },
        )
        last_n_publications_chain = SingleFilterChainBuilder.create_chain(
            llm=self._llm,
            system_prompt=DialRagPrompts.PREFILTER_SYSTEM_PROMPT_LAST_N_PUBLICATIONS,
            output_type=LastNPublicationsFilter,
        )
        chain = RunnableParallel(
            {
                "time_period": date_chain,
                "latest": latest_chain,
                "publication_types": publication_types_chain,
                "last_n_publications": last_n_publications_chain,
            }
        )
        return chain

    def _convert_llm_output_to_dial_rag_filter(
        self, llm_output: RagFilterLLMOutput, reference_date: date | None = None
    ) -> RagFilterDial | None:
        # fix hallucinations before creating the filter
        time_period_llm = llm_output.time_period.fix_llm_hallucinations()

        start_date = time_period_llm.parse_date(time_period_llm.start)
        end_date = time_period_llm.parse_date(time_period_llm.end)

        return self._dial_rag_prefilter_builder.create_prefilter(
            publication_types=llm_output.publication_types,
            start_date=start_date,
            end_date=end_date,
            is_latest=llm_output.is_latest,
            last_n_publications=llm_output.last_n_publications,
            reference_date=reference_date,
        )


class DialRagPrefilterBuilder:
    DEFAULT_DECODER_OF_LATEST = "-1y"

    def __init__(
        self,
        metadata: DialRagMetadata,
        pub_type_to_decoder_mapping: dict[str, str],
    ):
        self._metadata = metadata
        self._pub_type_to_decoder_mapping = pub_type_to_decoder_mapping

    def create_prefilter(
        self,
        publication_types: list[str],
        start_date: date | None,
        end_date: date | None,
        is_latest: bool,
        last_n_publications: int | None,
        reference_date: date | None,
    ) -> RagFilterDial | None:
        grounded_publication_types = self._ground_selected_publication_types(publication_types)
        filters = (
            self._create_latest_filters(grounded_publication_types, end_date, reference_date)
            if is_latest and not last_n_publications and not start_date
            else self._create_filters(grounded_publication_types, start_date, end_date)
        )
        params = dict()
        if filters:
            params["filters"] = filters
        if last_n_publications:
            params["top_n"] = TopNDocuments(
                sort_by=["publication_date"], order=SortOrder.desc, limit=last_n_publications
            )
        return RagFilterDial(**params) if params else None

    @staticmethod
    def _create_filters(
        grounded_publication_types: list[str],
        start_date: date | None,
        end_date: date | None,
    ) -> list[RagFilterDialSingle]:
        try:
            time_period = TimePeriodFilterDial(start=start_date, end=end_date)
        except ValueError as e:
            logger.error(f"Failed to build time period filter. Error: {e}")
            time_period = None
        if time_period and not grounded_publication_types:
            return [
                RagFilterDialSingle(
                    publication_type=None,
                    publication_date=time_period,
                )
            ]
        return [
            RagFilterDialSingle(
                publication_type=pub_type,
                publication_date=time_period,
            )
            for pub_type in grounded_publication_types
        ]

    def _create_latest_filters(
        self,
        grounded_publication_types: list[str],
        end_date: date | None,
        reference_date: date | None,
    ) -> list[RagFilterDialSingle]:
        if not grounded_publication_types:
            # NOTE: This is a special case where we need to return all publication types,
            # since `publication_type` is required to calculate the `latest` time period.
            # NOTE: even if LLM selected only hallucinations and no valid publication types,
            # it's okay to return all available publication types.
            grounded_publication_types = list(self._metadata.publication_types)
        filters = []
        for pub_type in grounded_publication_types:
            decoder = self._pub_type_to_decoder_mapping.get(
                pub_type, self.DEFAULT_DECODER_OF_LATEST
            )
            time_period = TimePeriodFilterDial.decode_latest_value(
                decoder=decoder, end_date=end_date, reference_date=reference_date
            )
            filters.append(
                RagFilterDialSingle(
                    publication_type=pub_type,
                    publication_date=time_period,
                )
            )
        return filters

    def _ground_selected_publication_types(self, publication_types: list[str]) -> list[str]:
        selected = set(publication_types)
        available = self._metadata.publication_types

        res = selected.intersection(available)
        hallucinations = selected.difference(res)

        if hallucinations:
            logger.warning(f"LLM hallucinated publication types: {hallucinations}")

        return list(res)

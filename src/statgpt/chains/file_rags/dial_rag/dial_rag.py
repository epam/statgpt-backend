from typing import Any

from aidial_sdk.chat_completion import Stage
from langchain_core.runnables import Runnable, RunnableLambda
from openai import APIError
from openai.types.chat import ChatCompletionUserMessageParam

from common.auth.auth_context import AuthContext
from common.config import multiline_logger as logger
from common.utils import MediaTypes
from common.utils.models import get_chat_model
from statgpt.chains.file_rags.base import BaseRAGFactory
from statgpt.chains.file_rags.dial_rag.metadata_loader import DialRagMetadataLoader
from statgpt.chains.file_rags.dial_rag.prefilter import PreFilterBuilder
from statgpt.chains.parameters import ChainParameters
from statgpt.config import StateVarsConfig
from statgpt.schemas import DialRagArtifact, DialRagState
from statgpt.schemas.file_rags.dial_rag import DialRagMetadata, PreFilterResponse
from statgpt.settings.dial_rag import dial_rag_settings
from statgpt.utils import OpenAiToDialStreamer, openai, replace_dial_url


class RAGMetadataError(Exception):
    pass


class DialRagAgentFactory(BaseRAGFactory):
    FIELD_PRE_FILTER = 'pre_filter'
    FIELD_ATTACHMENTS = 'attachments'
    FIELD_METADATA = 'metadata'
    FIELD_PRE_FILTER_DECODER_OF_LATEST = 'prefilter_decoder_of_latest'

    def _init_dial_rag_client(self, auth_context: AuthContext):
        nondefault_dial_rag_pgvector_endpoint = dial_rag_settings.pgvector_url
        nondefault_dial_rag_pgvector_api_key = dial_rag_settings.pgvector_api_key

        if nondefault_dial_rag_pgvector_endpoint and nondefault_dial_rag_pgvector_api_key:
            logger.info(
                f'Using non-default DIAL RAG PGVector endpoint: {nondefault_dial_rag_pgvector_endpoint}'
            )
            client = openai.get_async_client(
                api_key=nondefault_dial_rag_pgvector_api_key,
                azure_endpoint=nondefault_dial_rag_pgvector_endpoint,
            )
            return client

        client = openai.get_async_client(api_key=auth_context.api_key)

        return client

    @staticmethod
    async def _add_prefilter_to_stage_attachments(
        target: Stage, pre_filter_response: PreFilterResponse
    ) -> None:
        if pre_filter_response.user_friendly_error:
            target.append_content(f"\n\n{pre_filter_response.user_friendly_error}\n\n")

        if llm_output := pre_filter_response.llm_output:
            llm_filters_str = f"```json\n{llm_output.model_dump_json(indent=2)}\n```"
        else:
            llm_filters_str = 'Failed to build publications pre-filter'
        target.add_attachment(
            type=MediaTypes.MARKDOWN, title='Pre-filter, LLM output', data=llm_filters_str
        )

        if rag_filter := pre_filter_response.rag_filter:
            rag_filter_json = rag_filter.model_dump_json(indent=2, exclude_none=True)
            rag_filter_str = f"```json\n{rag_filter_json}\n```"
        else:
            rag_filter_str = 'No filter applied'
        target.add_attachment(
            type=MediaTypes.MARKDOWN, title='Pre-filter, final', data=rag_filter_str
        )

    async def _run_prefilter_nonsafe(
        self, auth_context: AuthContext, query: str
    ) -> tuple[PreFilterResponse, DialRagMetadata]:
        try:
            metadata_loader = DialRagMetadataLoader.create_for_local_or_remote(
                auth_context=auth_context,
                metadata_endpoint=self._tool_config.details.metadata_endpoint,
            )
            metadata_resp = await metadata_loader.load()
            metadata = DialRagMetadata.from_response(metadata_resp)
        except Exception as e:
            raise RAGMetadataError from e

        llm = get_chat_model(
            api_key=auth_context.api_key,
            model_config=self._tool_config.details.prefilter_llm_model_config,
        )
        logger.info(
            f"{self.__class__.__name__} using LLM model: "
            f"{self._tool_config.details.prefilter_llm_model_config.deployment.deployment_id}"
        )
        decoder_of_latest_mapping = self._tool_config.details.decoder_of_latest
        pre_filter_builder = PreFilterBuilder(
            llm=llm, metadata=metadata, pub_type_to_decoder_mapping=decoder_of_latest_mapping
        )

        pre_filter_response = await pre_filter_builder.build_filter_from_query(query=query)
        logger.info(f'Built publication filter: {pre_filter_response!r}')

        return pre_filter_response, metadata

    async def _run_prefilter(
        self, auth_context: AuthContext, query: str, target: Stage
    ) -> tuple[PreFilterResponse, DialRagMetadata | None]:

        def _format_exception_w_cause(exc: Exception) -> str:
            return f'error {repr(exc)}. cause: {repr(exc.__cause__)}'

        try:
            pre_filter_response, metadata = await self._run_prefilter_nonsafe(
                auth_context=auth_context, query=query
            )
        except RAGMetadataError as e:
            logger.exception(e)
            pre_filter_response = PreFilterResponse(
                user_friendly_error="Failed to retrieve publications metadata from RAG - can't build publications pre-filter",
                detailed_error=_format_exception_w_cause(e),
                llm_output=None,
                rag_filter=None,
            )
            metadata = None
        except Exception as e:
            logger.exception(e)
            pre_filter_response = PreFilterResponse(
                user_friendly_error="Failed to build publications pre-filter",
                detailed_error=_format_exception_w_cause(e),
                llm_output=None,
                rag_filter=None,
            )
            metadata = None

        await self._add_prefilter_to_stage_attachments(target, pre_filter_response)

        return pre_filter_response, metadata

    def _append_attachments(self, target: Stage, attachments: list[dict[str, Any]]) -> None:
        attachment_url_override = self._tool_config.details.get_attachment_url_override()

        for attachment in attachments:
            reference_url = attachment.get('reference_url')
            if attachment_url_override and reference_url:
                reference_url = replace_dial_url(reference_url, attachment_url_override)

            target.add_attachment(
                type=attachment.get('type'),
                title=attachment.get('title'),
                data=attachment.get('data'),
                url=attachment.get('url'),
                reference_url=reference_url,
                reference_type=attachment.get('reference_type'),
            )

    async def _stream_response(self, inputs: dict) -> dict:
        logger.info(f'{type(self).__name__}._stream_response()')

        auth_context = ChainParameters.get_auth_context(inputs)
        target = ChainParameters.get_target(inputs)
        choice = ChainParameters.get_choice(inputs)
        query = ChainParameters.get_query(inputs)

        target_prefilter = ChainParameters.get_target_prefilter(inputs)
        if target_prefilter is not None:
            logger.info(
                'received target prefilter - will ignore building prefilter from user query. '
                f'target_prefilter: {target_prefilter}'
            )
            pre_filter_response = PreFilterResponse(
                llm_output=None,  # not used in RAG eval, since we pass prefilter directly
                rag_filter=target_prefilter,
            )
            metadata = None  # not used as well
        else:
            logger.info(f'building prefilter from user query: "{query}"')
            pre_filter_response, metadata = await self._run_prefilter(
                auth_context=auth_context, query=query, target=target
            )

        inputs[self.FIELD_PRE_FILTER] = pre_filter_response
        inputs[self.FIELD_METADATA] = metadata
        inputs[self.FIELD_PRE_FILTER_DECODER_OF_LATEST] = (
            self._tool_config.details.decoder_of_latest
        )

        state = ChainParameters.get_state(inputs)
        skip = state.get(StateVarsConfig.CMD_RAG_PREFILTER_ONLY, False)

        if skip:
            inputs[self.FIELD_RESPONSE] = (
                f'<call to RAG was skipped for debug purposes>\n\nquery: "{query}"\n\n---'
            )
            # NOTE: set to "RAG" but actually there was no any response
            inputs[self.FIELD_ANSWERED_BY] = 'RAG'
            target.append_content(inputs[self.FIELD_RESPONSE])
            return inputs

        # call dial RAG

        prefilter_dict = None if (f := pre_filter_response.rag_filter) is None else f.as_dial_dict()
        inputs_to_log = {'query': query, 'prefilter': prefilter_dict}
        logger.info(f'calling DIAL RAG with following inputs: {inputs_to_log}')

        configuration_params = (
            None if prefilter_dict is None else {"custom_fields": {"configuration": prefilter_dict}}
        )

        dial_rag_client = self._init_dial_rag_client(auth_context)
        rag_stream = await dial_rag_client.chat.completions.create(
            model=self._tool_config.details.deployment_id,
            stream=True,
            messages=[ChatCompletionUserMessageParam(role='user', content=query)],
            extra_body=configuration_params,
        )

        dial_streamer = OpenAiToDialStreamer(
            target,
            choice,
            deployment=self._tool_config.details.deployment_id,
            stream_content=False,
            show_debug_stages=state.get(StateVarsConfig.SHOW_DEBUG_STAGES, False),
            stages_config=self._tool_config.details.stages_config,
        )
        with dial_streamer:
            try:
                async for chunk in rag_stream:
                    dial_streamer.send_chunk(chunk)
            except APIError as e:
                logger.exception(e)

            # NOTE: append '---' to the end to create space between text and attachments
            if dial_streamer.attachments:
                target.append_content(
                    "Answer using the information found by RAG.\n\n---\n\n"
                    f'### Query\n\n{query}\n\n'
                    f"### Response\n\n{dial_streamer.content}"
                    "\n\n---"
                )
                self._append_attachments(target, dial_streamer.attachments)
                inputs[self.FIELD_RESPONSE] = dial_streamer.content_with_attachments_metadata
                inputs[self.FIELD_ANSWERED_BY] = 'RAG'
                inputs[self.FIELD_ATTACHMENTS] = dial_streamer.attachments
            else:
                tool_name = self._tool_config.name.replace('_', ' ')
                msg = (
                    f'{tool_name} was unable to find the relevant data for the query: "{query}"'
                    '\n\n---'
                )
                logger.info(
                    f"{tool_name} was unable to find the relevant data for the query: {query}\nOriginal response: {dial_streamer.content_with_attachments_metadata}"
                )
                target.append_content(msg)
                inputs[self.FIELD_RESPONSE] = msg
                inputs[self.FIELD_ANSWERED_BY] = 'LLM'

        return inputs

    @classmethod
    def _set_tool_state(cls, inputs: dict) -> dict:
        agent_state = DialRagState(**inputs)
        inputs[cls.FIELD_ARTIFACT] = DialRagArtifact(state=agent_state)

        return inputs

    async def create_chain(self) -> Runnable:
        return RunnableLambda(self._stream_response) | self._set_tool_state

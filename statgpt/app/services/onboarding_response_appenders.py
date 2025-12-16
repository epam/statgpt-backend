import json
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, overload

from aidial_sdk.chat_completion import Choice

from statgpt.app.chains.utils import time_period_utils
from statgpt.common.auth.auth_context import AuthContext
from statgpt.common.schemas.onboarding import PredefinedDataQueryResponse, PredefinedTextResponse
from statgpt.common.schemas.query import JsonQuery, JsonQueryMetadata, JsonQueryWithMetadata
from statgpt.common.utils import MediaTypes

from .chat_facade import ChannelServiceFacade

_T = TypeVar("_T", PredefinedTextResponse, PredefinedDataQueryResponse, None)

_log = logging.getLogger(__name__)


class BaseResponseAppender(Generic[_T], ABC):

    def __init__(self, response: _T) -> None:
        self._response: _T = response

    @abstractmethod
    async def append_to_response(
        self,
        choice: Choice,
        channel_service: ChannelServiceFacade,
        auth_context: AuthContext,
    ) -> None:
        pass


class PredefinedTextResponseAppender(BaseResponseAppender[PredefinedTextResponse]):

    def __init__(self, response: PredefinedTextResponse) -> None:
        super().__init__(response)

    async def append_to_response(
        self,
        choice: Choice,
        channel_service: ChannelServiceFacade,
        auth_context: AuthContext,
    ) -> None:
        choice.append_content(self._response.text)


class PredefinedDataQueryResponseAppender(BaseResponseAppender[PredefinedDataQueryResponse]):

    def __init__(self, response: PredefinedDataQueryResponse) -> None:
        super().__init__(response)

    async def _append_json_query_attachment(
        self,
        choice: Choice,
        channel_service: ChannelServiceFacade,
        auth_context: AuthContext,
    ) -> None:
        urn = self._response.query.urn

        dataset = await channel_service.get_dataset_by_source_id(
            auth_context=auth_context,
            dataset_id=urn,
        )
        if dataset is None:
            _log.warning("Dataset with id %s not found", urn)
            return
        json_query_metadata = JsonQueryMetadata(
            country_dimension=dataset.config.country_dimension,
            indicator_dimensions=dataset.config.indicator_dimensions,
            dataset_url=dataset.dataset_url,
        )
        json_query = JsonQueryWithMetadata.from_query(
            query=self._get_relative_time_period_aware_query(self._response.query),
            metadata=json_query_metadata,
        ).model_dump(by_alias=True)
        json_query_content = json.dumps(json_query)
        choice.add_attachment(
            type=MediaTypes.JSON,
            title=f"Query (JSON): {urn}",
            data=json_query_content,
        )

    async def append_to_response(
        self,
        choice: Choice,
        channel_service: ChannelServiceFacade,
        auth_context: AuthContext,
    ) -> None:
        await self._append_json_query_attachment(choice, channel_service, auth_context)
        choice.append_content(self._response.text)

    def _get_relative_time_period_aware_query(self, query: JsonQuery) -> JsonQuery:
        for filter in query.filters:
            if filter.component_code == "TIME_PERIOD":
                filter.values = [
                    time_period_utils.get_relative_aware_time_period(value)
                    for value in filter.values
                ]
        return query


class NoOpResponseAppender(BaseResponseAppender[None]):

    def __init__(self, response: None) -> None:
        super().__init__(response)

    async def append_to_response(
        self,
        choice: Choice,
        channel_service: ChannelServiceFacade,
        auth_context: AuthContext,
    ) -> None:
        pass


class ResponseAppenderFactory:

    _appender_map = {
        PredefinedTextResponse: PredefinedTextResponseAppender,
        PredefinedDataQueryResponse: PredefinedDataQueryResponseAppender,
    }

    @staticmethod
    @overload
    def get_appender(response: None) -> NoOpResponseAppender: ...

    @staticmethod
    @overload
    def get_appender(response: PredefinedTextResponse) -> PredefinedTextResponseAppender: ...

    @staticmethod
    @overload
    def get_appender(
        response: PredefinedDataQueryResponse,
    ) -> PredefinedDataQueryResponseAppender: ...

    @staticmethod
    def get_appender(
        response: PredefinedTextResponse | PredefinedDataQueryResponse | None,
    ) -> BaseResponseAppender:
        if response is None:
            return NoOpResponseAppender(response)

        appender_class = ResponseAppenderFactory._appender_map.get(type(response))
        if appender_class is None:
            raise ValueError(f"Unsupported response type: {type(response)}")

        return appender_class(response)

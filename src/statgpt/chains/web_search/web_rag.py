import re
from enum import StrEnum
from typing import Any

from pydantic import Field, create_model

from common.config import multiline_logger as logger
from common.schemas import ChannelConfig, ToolTypes
from common.schemas import WebSearchTool as WebSearchToolConfig
from statgpt.chains.parameters import ChainParameters
from statgpt.chains.tools import StatGptTool, ToolArgs
from statgpt.schemas import ToolArtifact, ToolMessageState

from .response_producer import RagResponseProducer, UrlOnlyResponseProducer


class BaseWebSearchArgs(ToolArgs):
    query: str = Field(
        description=(
            "Natural language query optimized for ai web search."
            " The value of this field cannot include the `site:<domain>` operator,"
            " the corresponding argument should be used instead if necessary."
        )
    )
    # selected_domains — defined dynamically in the code below


class WebSearchTool(StatGptTool[WebSearchToolConfig], tool_type=ToolTypes.WEB_SEARCH):
    def __init__(self, tool_config: WebSearchToolConfig, channel_config: ChannelConfig, **kwargs):
        super().__init__(tool_config, channel_config, **kwargs)

        kwargs = dict(
            deployment_id=tool_config.details.deployment_id,
            stages_config=tool_config.details.stages_config,
        )
        if tool_config.details.urls_only:
            self._response_producer = UrlOnlyResponseProducer(**kwargs)
        else:
            self._response_producer = RagResponseProducer(**kwargs)

    @classmethod
    def get_args_schema(cls, tool_config: WebSearchToolConfig) -> type[BaseWebSearchArgs]:
        """Return the schema for the arguments that this tool accepts."""

        other_fields = {}

        if domains_config := tool_config.details.domains:
            domain_enum = StrEnum(
                'DomainEnum',
                [(cls._to_enum_name(d), d) for d in domains_config.allowed_values],
            )
            other_fields[domains_config.field_name] = (
                list[domain_enum],
                Field(default_factory=lambda: [], description=domains_config.field_description),
            )

        web_search_args_cls = create_model(
            'WebSearchArgs', **other_fields, __base__=BaseWebSearchArgs
        )

        return web_search_args_cls

    @staticmethod
    def _to_enum_name(domain: str) -> str:
        return domain.upper().replace(".", "_")

    @property
    def allowed_domains(self) -> list[str] | None:
        if domains_config := self._tool_config.details.domains:
            return domains_config.allowed_values
        # If the domains field is not configured, all domains are allowed:
        return None

    def _extract_domains(self, kwargs: dict[str, Any]) -> list[str] | None:
        if domains_config := self._tool_config.details.domains:
            selected_domains = kwargs.get(domains_config.field_name)
            if not selected_domains:
                selected_domains = domains_config.allowed_values
            return selected_domains
        # If the domains field is not configured, all domains are allowed:
        return None

    def _validate_query(self, query: str) -> str | None:
        """Check if the query contains prohibited domains. All domains except the allowed ones are prohibited."""

        if self.allowed_domains is None:
            return None  # No domain restrictions

        prohibited_domains = []
        for d in re.findall(r"site:(\S+)", query):
            if d in self.allowed_domains:
                logger.info(
                    f"Domain {d} is allowed, but it is specified in the query. Removing it."
                )
                query = query.replace(f"site:{d}", "")
            else:
                prohibited_domains.append(d)

        if not prohibited_domains:
            return None

        return f"Query contains prohibited domains: {prohibited_domains}"

    @staticmethod
    def _prepare_query(query: str, domains: list[str]) -> str:
        if not domains:
            return query
        domains_filter = " | ".join([f"site:{domain}" for domain in domains])
        return f"!web {query} ({domains_filter})\n!rag {query}"

    async def _arun(self, inputs: dict, query: str, **kwargs) -> tuple[str, ToolArtifact]:
        target = ChainParameters.get_target(inputs)

        if error := self._validate_query(query):
            target.append_content(error)
            return error, ToolArtifact(state=ToolMessageState(type=self.tool_type))

        selected_domains = self._extract_domains(kwargs)
        prepared_query = self._prepare_query(query, selected_domains)
        logger.info(f"Full query for web search: {prepared_query!r}")

        str_response = await self._response_producer.run(inputs=inputs, query=prepared_query)
        target.append_content(str_response)

        artifact = ToolArtifact(state=ToolMessageState(type=self.tool_type))
        return str_response, artifact

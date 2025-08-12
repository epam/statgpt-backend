from collections.abc import Iterable
from typing import Any

from elasticsearch import AsyncElasticsearch, helpers
from pydantic import BaseModel, ConfigDict, Field

from common.settings.elastic import ElasticSearchSettings


class HitTotal(BaseModel):
    """Metadata about the number of matching documents."""

    value: int = Field(description="Total number of matching documents.")
    relation: str = Field(
        description="Indicates whether the number of matching documents in the value parameter is accurate or a lower bound."
    )

    model_config = ConfigDict(populate_by_name=True)


class Hit(BaseModel):
    index: str = Field(
        alias='_index', description="Name of the index containing the returned document."
    )
    id: str = Field(
        alias='_id',
        description="Unique identifier for the returned document. This ID is only unique within the returned index.",
    )
    score: float = Field(
        alias='_score',
        description="Positive 32-bit floating point number used to determine the relevance of the returned document.",
    )
    source: dict = Field(
        alias='_source',
        description="Original JSON body passed for the document at index time.",
    )

    highlight: dict | None = Field(
        default=None,
        description=(
            "Contains highlighted snippets from the returned document."
            " This field is only present if highlighting is requested."
        ),
    )

    # NOTE: Add this field if needed:
    # fields

    model_config = ConfigDict(populate_by_name=True)


class Hits(BaseModel):
    total: HitTotal
    max_score: float | None = Field(
        description="Highest returned document _score. This value is `None` for requests that do not sort by _score."
    )
    hits: list[Hit]

    model_config = ConfigDict(populate_by_name=True)


class SearchResult(BaseModel):
    took: int = Field(description="Milliseconds it took Elasticsearch to execute the request.")
    timed_out: bool = Field(
        description="If true, the request timed out before completion; returned results may be partial or empty."
    )
    hits: Hits = Field(description="Contains returned documents and metadata.")

    aggregations: dict | None = Field(
        default=None, description="Contains aggregation results if aggregations were requested."
    )

    # NOTE: Add this field if needed:
    # scroll_id: str| None = Field(alias="_scroll_id")
    # shards: dict = Field(alias="_shards")

    model_config = ConfigDict(populate_by_name=True)


class Token(BaseModel):
    token: str
    start_offset: int
    end_offset: int
    type: str
    position: int


class ElasticIndex:
    def __init__(
        self,
        client: AsyncElasticsearch,
        name: str,
        settings: ElasticSearchSettings,
    ):
        self._client = client
        self._name = name
        self._settings = settings

    @property
    def name(self) -> str:
        """Name of the index."""
        return self._name

    async def exists(self) -> bool:
        return bool(await self._client.indices.exists(index=self.name))

    async def create(self):
        await self._client.indices.create(
            index=self._name,
            settings=self._settings.index_settings,
        )

    async def add(self, document: dict[str, str]) -> None:
        """Adds a JSON document to the index and makes it searchable.
        If the document already exists, the request updates the document and increments its version.
        """

        await self._client.index(index=self.name, document=document, id=document["id"])

    async def add_bulk(self, documents: Iterable[dict[str, str]]) -> None:
        """Adds multiple JSON documents to the index in several requests.
        See elastic documentation for default `chunk_size` and `max_chunk_bytes` values.
        """

        actions = ({"_index": self.name, "_id": doc["id"], "_source": doc} for doc in documents)
        await helpers.async_bulk(self._client, actions)

    async def search(self, *, query: dict, from_: int = 0, size: int, **kwargs) -> SearchResult:
        """Returns search hits that match the query defined in the request."""

        result = await self._client.search(
            index=self.name, query=query, from_=from_, size=size, **kwargs
        )
        return SearchResult.model_validate(result.body)

    async def analyze(self, *, text: str) -> list[Token]:
        """Performs analysis on a text string and returns the resulting tokens."""

        result = await self._client.indices.analyze(index=self.name, text=text)
        return [Token.model_validate(token) for token in result["tokens"]]


class ElasticSearchFactory:
    _client: AsyncElasticsearch | None = None
    _indexes: dict[str, ElasticIndex] = {}
    _settings: ElasticSearchSettings = ElasticSearchSettings()

    @classmethod
    async def get_client(cls) -> AsyncElasticsearch:
        if cls._client is None:
            cls._client = await cls.create_client()

        return cls._client

    @classmethod
    async def create_client(cls) -> AsyncElasticsearch:
        kwargs: dict[str, Any] = dict(
            hosts=cls._settings.connection_string,
            timeout=cls._settings.timeout,
            max_retries=cls._settings.max_retries,
            retry_on_timeout=cls._settings.retry_on_timeout,
        )
        if cls._settings.auth_user and cls._settings.auth_password:
            user = cls._settings.auth_user.strip()
            password = cls._settings.auth_password.get_secret_value().strip()
            kwargs["basic_auth"] = (user, password)

        es_client = AsyncElasticsearch(**kwargs)

        return es_client

    @classmethod
    async def get_index(cls, name: str, allow_creation: bool = False) -> ElasticIndex:
        if name not in cls._indexes:
            client = await cls.get_client()
            index = ElasticIndex(client, name, cls._settings)

            if not await index.exists():
                if allow_creation:
                    await index.create()
                else:
                    raise RuntimeError(f"Index '{name}' does not exist.")

            cls._indexes[name] = index

        return cls._indexes[name]

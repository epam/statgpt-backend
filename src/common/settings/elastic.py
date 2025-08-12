from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_ES_INDEX_DEFAULT_SETTINGS = {
    "analysis": {
        "filter": {"stop_filter": {"type": "stop", "ignore_case": True, "stopwords": "_english_"}},
        "analyzer": {
            "default": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "stop_filter", "stemmer"],
            },
            "default_search": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "stop_filter", "stemmer"],
            },
        },
    }
}


class ElasticSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="elastic_")

    # client configuration
    connection_string: str | None = Field(None, description="Elasticsearch connection string")
    auth_user: str = Field(default='', description="Elasticsearch authentication user")
    auth_password: SecretStr = Field(
        default=SecretStr(''), description="Elasticsearch authentication password"
    )
    timeout: int = Field(30, description="Elasticsearch connection timeout")
    max_retries: int = Field(3, description="Elasticsearch connection max retries")
    retry_on_timeout: bool = Field(True, description="Elasticsearch connection retry on timeout")

    indicators_index: str | None = Field(None, description="Elasticsearch index for indicators")
    matching_index: str | None = Field(None, description="Elasticsearch index for matching")

    index_settings: dict = Field(
        default=_ES_INDEX_DEFAULT_SETTINGS,
        description="Default settings for Elasticsearch indices",
    )

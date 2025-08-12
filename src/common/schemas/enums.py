from enum import StrEnum


class PreprocessingStatusEnum(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobType(StrEnum):
    EXPORT = "EXPORT"
    IMPORT = "IMPORT"


class ToolTypes(StrEnum):
    AVAILABLE_DATASETS = "AVAILABLE_DATASETS"
    AVAILABLE_PUBLICATIONS = "AVAILABLE_PUBLICATIONS"
    AVAILABLE_TERMS = "AVAILABLE_TERMS"
    DATA_QUERY = "DATA_QUERY"
    FILE_RAG = "FILE_RAG"
    PLAIN_CONTENT = "PLAIN_CONTENT"
    TERM_DEFINITIONS = "TERM_DEFINITIONS"
    WEB_SEARCH = "WEB_SEARCH"
    WEB_SEARCH_AGENT = "WEB_SEARCH_AGENT"


class RAGVersion(StrEnum):
    DIAL = "DIAL"
    """DIAL RAG PgVector"""


class DecoderOfLatestEnum(StrEnum):
    """Function to create a time range corresponding to "latest" for a given publication type."""

    LAST_YEAR = "last_year"
    # LAST_PUBLICATION = "last_publication"


class DataQueryVersion(StrEnum):
    v1 = "v1"
    v2 = "v2"


class IndexerVersion(StrEnum):
    semantic = "semantic"
    hybrid = "hybrid"


class IndicatorSelectionVersion(StrEnum):
    hybrid = "hybrid"
    semantic_v1 = "semantic_v1"
    semantic_v2 = "semantic_v2"
    semantic_v3 = "semantic_v3"
    semantic_v4 = "semantic_v4"

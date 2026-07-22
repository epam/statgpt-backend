from enum import StrEnum


class PreprocessingStatusEnum(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    QUEUED = "QUEUED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

    @classmethod
    def final_statuses(cls) -> list["PreprocessingStatusEnum"]:
        """Return a list of statuses that indicate a job has reached a final state
        (either completed successfully or failed)."""
        return [cls.COMPLETED, cls.FAILED]


class JobType(StrEnum):
    EXPORT = "EXPORT"
    IMPORT = "IMPORT"


class ExportScope(StrEnum):
    FULL = "full"
    CONFIGS = "configs"
    INDEXES = "indexes"
    DIAL_FILES = "dial_files"

    def includes_configs(self) -> bool:
        """Check if this scope includes configuration export."""
        return self is ExportScope.CONFIGS or self is ExportScope.FULL

    def includes_indexes(self) -> bool:
        """Check if this scope includes indexes export."""
        return self is ExportScope.INDEXES or self is ExportScope.FULL

    def includes_dial_files(self) -> bool:
        """Check if this scope includes DIAL files export."""
        return self is ExportScope.DIAL_FILES or self is ExportScope.FULL


class ToolTypes(StrEnum):
    AVAILABLE_DATASETS = "AVAILABLE_DATASETS"
    DATASETS_METADATA = "DATASETS_METADATA"
    DATASET_STRUCTURE = "DATASET_STRUCTURE"
    AVAILABLE_PUBLICATIONS = "AVAILABLE_PUBLICATIONS"
    AVAILABLE_TERMS = "AVAILABLE_TERMS"
    DATA_QUERY = "DATA_QUERY"
    FILE_RAG = "FILE_RAG"
    PLAIN_CONTENT = "PLAIN_CONTENT"
    SDMX_QUERY_APP = "SDMX_QUERY_APP"
    TERM_DEFINITIONS = "TERM_DEFINITIONS"
    WEB_SEARCH = "WEB_SEARCH"
    WEB_SEARCH_AGENT = "WEB_SEARCH_AGENT"
    DEEP_RESEARCH = "DEEP_RESEARCH"


class McpResourceTypes(StrEnum):
    PROXIED = "PROXIED"
    """A resource whose content is proxied verbatim from an external HTTP endpoint."""


class RAGVersion(StrEnum):
    DIAL = "DIAL"
    """DIAL RAG PgVector"""


class DecoderOfLatestEnum(StrEnum):
    """Function to create a time range corresponding to "latest" for a given publication type."""

    LAST_YEAR = "last_year"
    # LAST_PUBLICATION = "last_publication"


class IndexerVersion(StrEnum):
    semantic = "semantic"
    hybrid = "hybrid"


class IndicatorSelectionVersion(StrEnum):
    hybrid = "hybrid"
    semantic_v1 = "semantic_v1"
    semantic_v2 = "semantic_v2"
    semantic_v3 = "semantic_v3"
    semantic_v4 = "semantic_v4"


class TimePeriodStrategy(StrEnum):
    BEFORE = "BEFORE"
    AFTER = "AFTER"


class SpecialDimensionsProcessorType(StrEnum):
    LHCL = "large_hierarchical_codelist"


class AvailableDatasetsVersion(StrEnum):
    short = "short"
    full = "full"


class AvailableDatasetsHeaderFormat(StrEnum):
    totals = "totals"
    agencies = "agencies"


_LANGUAGE_NAMES = {
    "en": "English",
    "uk": "Ukrainian",
}


class LocaleEnum(StrEnum):
    EN = "en"
    UK = "uk"

    def get_language_name(self) -> str:
        """Return the full language name for this locale."""
        return _LANGUAGE_NAMES[self]


class DataRequestStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIALLY_FAILED = "PARTIALLY_FAILED"


class DataParsingStatus(StrEnum):
    NA = "NA"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIALLY_FAILED = "PARTIALLY_FAILED"


class ChannelIndexStatusScope(StrEnum):
    FULL = "full"
    LATEST_COMPLETED_VERSIONS = "latest_completed_versions"


class ChannelDatasetUpdateStatus(StrEnum):
    AUTO_UPDATED = "auto_updated"
    NEEDS_REINDEX = "needs_reindex"
    NO_VERSION = "no_version"
    INDEXING_IN_PROGRESS = "indexing_in_progress"


class AuditEntityType(StrEnum):
    CHANNEL = "channel"
    DATASET = "dataset"
    DATA_SOURCE = "data_source"
    IMPORT_JOB = "import_job"


class AuditActionType(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class AutoUpdateResult(StrEnum):
    """Result of an auto-update job execution."""

    NO_COMPLETED_VERSION = "NO_COMPLETED_VERSION"
    """No base version to update from."""

    NO_CHANGES = "NO_CHANGES"
    """Dataset is up to date."""

    CONFIG_UPDATED = "CONFIG_UPDATED"
    """Config updated (e.g., URN version changed) but data unchanged, no reindexing needed."""

    CONFIG_INCOMPATIBLE = "CONFIG_INCOMPATIBLE"
    """New version found but config not applicable."""

    REINDEX_TRIGGERED = "REINDEX_TRIGGERED"
    """Reindexing started (check created_version.preprocessing_status for completion)."""

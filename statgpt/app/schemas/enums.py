from enum import StrEnum


class DataQueryStatus(StrEnum):
    """Outcome of the data query pipeline, tagging which branch produced the response.

    Surfaced in the Data Query tool's MCP structured content so callers can act on the
    result programmatically instead of parsing the human-readable text.
    """

    DATA_AVAILABLE = "data_available"
    NO_DATA = "no_data"
    DATASET_SELECTION_REQUIRED = "dataset_selection_required"
    INVALID_TIME_PERIOD = "invalid_time_period"
    MISSING_DIMENSIONS = "missing_dimensions"
    NOT_EXECUTED = "not_executed"

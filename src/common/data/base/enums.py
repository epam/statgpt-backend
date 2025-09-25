from enum import StrEnum


class EntityType(StrEnum):
    DATA_SOURCE = "DataSource"
    DATA_SET = "DataSet"
    DIMENSION = "Dimension"
    ATTRIBUTE = "Attribute"
    INDICATOR = "Indicator"
    CATEGORY = "Category"
    OTHER = "Other"


class AttributeType(StrEnum):
    CATEGORY = "category"
    STRING = "string"


class DimensionType(StrEnum):
    CATEGORY = "category"
    DATETIME = "datetime"


class QueryOperator(StrEnum):
    """
    Enum representing different query operators for filtering data.
    """

    ALL = "all"
    """Special operator that matches all values"""

    EQUALS = "equals"
    """Matches the exact value"""

    NOT_EQUALS = "not_equals"
    """Matches all values except the exact value"""

    GREATER_THAN = "greater_than"
    """Matches values greater than the given value"""

    LESS_THAN = "less_than"
    """Matches values less than the given value"""

    GREATER_THAN_OR_EQUALS = "greater_than_or_equals"
    """Matches values greater than or equal to the given value"""

    LESS_THAN_OR_EQUALS = "less_than_or_equals"
    """Matches values less than or equal to the given value"""

    IN = "in"
    """Matches values in the given list"""

    BETWEEN = "between"
    """Matches values between the given range"""

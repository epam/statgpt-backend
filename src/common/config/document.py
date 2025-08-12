class IndicatorDocumentMetadataFields:
    """Field names in indicator document metadata"""

    DATA_SOURCE_ID = 'data_source_id'
    INDICATORS_METADATA = 'indicators'


class DimensionValueDocumentMetadataFields:
    """Field names in dimension value document metadata"""

    DIMENSION_ID = 'dimension_id'
    DATA_SOURCE_ID = 'data_source_id'


class VectorStoreMetadataFields:

    DOCUMENT_ID = 'document_id'
    TABLE_NAME = 'table_name'

    __ALL__ = [DOCUMENT_ID, TABLE_NAME]

from .dial import DialConfig
from .document import (
    DimensionValueDocumentMetadataFields,
    IndicatorDocumentMetadataFields,
    VectorStoreMetadataFields,
)
from .embeddings import EmbeddingsConfig
from .langchain_config import LangChainConfig
from .llm_models import LLMModelsConfig
from .logging import LoggingConfig, logger, multiline_logger
from .sdmx import SdmxConfig
from .versions import Versions


def _add_types():
    from mimetypes import add_type

    _TYPES_TO_ADD = (
        ("application/x-yaml", ".yaml"),
        ("application/x-yaml", ".yml"),
    )
    for mimetype, extension in _TYPES_TO_ADD:
        add_type(mimetype, extension)


_add_types()

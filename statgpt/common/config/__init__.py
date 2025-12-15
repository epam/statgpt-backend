from .llm_models import EmbeddingModelsEnum, LLMModelsEnum
from .logging import LoggingConfig, logger, multiline_logger
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

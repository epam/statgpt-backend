from typing import Any

from langchain_core.documents import Document
from pydantic import Field


class EmbeddedDocument(Document):
    """Document pydantic model with embeddings"""

    embeddings: list[float] | None = Field(
        default=None, description="Embeddings of the document. Can be None to save RAM space."
    )

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass `page_content` in as positional or named arg.

        Additionally, `metadata` and `embeddings` can be passed in as named args.
        """
        # my-py is complaining that page_content is not defined on the base class.
        # Here, we're relying on pydantic base class to handle the validation.
        super().__init__(page_content=page_content, **kwargs)  # type: ignore[call-arg]

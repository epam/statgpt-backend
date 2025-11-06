import uuid
from typing import Any

from langchain_core.documents import Document
from pydantic import Field


class EmbeddedDocument(Document):
    """LangChain Document (pydantic) model with embeddings"""

    embeddings: list[float] = Field(description="Embeddings of the document.")

    def init(self, page_content: str, **kwargs: Any) -> None:
        """Pass `page_content` in as positional or named arg.

        Additionally, `metadata` and `embeddings` can be passed in as named args.
        """
        # my-py is complaining that page_content is not defined on the base class.
        # Here, we're relying on pydantic base class to handle the validation.
        super().__init__(page_content=page_content, **kwargs)  # type: ignore[call-arg]


class VectorStoreDocument(Document):
    """LangChain Document (pydantic) model with additional fields from Vector Store"""

    table_name: str = Field(description="Name of the table where the document is stored.")
    document_id: int = Field(description="ID of the document in the database.")
    dataset_id: uuid.UUID = Field(description="UUID of the dataset the document belongs to.")
    version_id: int = Field(description="ID of the version the document belongs to.")

    def init(self, page_content: str, **kwargs: Any) -> None:
        """Pass `page_content` in as positional or named arg.

        Additionally, `metadata` and `embeddings` can be passed in as named args.
        """
        # my-py is complaining that page_content is not defined on the base class.
        # Here, we're relying on pydantic base class to handle the validation.
        super().__init__(page_content=page_content, **kwargs)  # type: ignore[call-arg]


class ScoredVectorStoreDocument(VectorStoreDocument):
    score: float = Field(description="Similarity score of the document.")

    def init(self, page_content: str, **kwargs: Any) -> None:
        """Pass `page_content` in as positional or named arg.

        Additionally, `metadata` and `embeddings` can be passed in as named args.
        """
        # my-py is complaining that page_content is not defined on the base class.
        # Here, we're relying on pydantic base class to handle the validation.
        super().__init__(page_content=page_content, **kwargs)  # type: ignore[call-arg]

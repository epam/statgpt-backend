import typing as t

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

from common.auth.auth_context import AuthContext
from common.utils.models import get_embeddings_model


class EmbeddingModel(BaseModel):
    name: str = Field(description="The name of the model")
    model: Embeddings = Field(description="The embeddings model")
    embedding_length: int = Field(description="The length of the embeddings")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EmbeddingModels:

    _AZURE_OPENAI_MODELS: set[str] = {"text-embedding-ada-002", "text-embedding-3-large"}

    def __init__(self):
        self._cache = {}

    def _get_az_openai_embeddings(
        self, model_name: str, auth_context: AuthContext
    ) -> t.Optional[EmbeddingModel]:
        if model_name in self._AZURE_OPENAI_MODELS:
            embeddings = get_embeddings_model(auth_context.api_key, model_name)
            length = len(embeddings.embed_query(""))
            embedding_model = EmbeddingModel(
                name=model_name,
                model=embeddings,
                embedding_length=length,
            )
            return embedding_model
        return None

    def get(self, model_name: str, auth_context: AuthContext) -> EmbeddingModel:
        if model := self._get_az_openai_embeddings(model_name, auth_context):
            return model
        raise ValueError(f"Unknown model name: {model_name}")

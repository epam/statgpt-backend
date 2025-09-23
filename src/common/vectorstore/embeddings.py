import typing as t

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

from common.auth.auth_context import AuthContext
from common.config import EmbeddingModelsEnum
from common.schemas import EmbeddingsModelConfig
from common.utils.models import get_embeddings_model


class EmbeddingModel(BaseModel):
    name: str = Field(description="The name of the model")
    model: Embeddings = Field(description="The embeddings model")
    embedding_length: int = Field(description="The length of the embeddings")
    is_normalized_to_one: bool = Field(
        description="Whether the embeddings are normalized to one",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EmbeddingModels:

    _AZURE_OPENAI_MODELS: set[str] = {"text-embedding-ada-002", "text-embedding-3-large"}

    length_cache: t.Dict[str, int]

    def __init__(self):
        self.length_cache = {}

    async def _get_embedding_model_length(
        self, model_name: str, model: Embeddings, sample_text: str = ""
    ) -> int:
        if model_name in self.length_cache:
            return self.length_cache[model_name]

        embedding_length = len(await model.aembed_query(sample_text))
        self.length_cache[model_name] = embedding_length
        return embedding_length

    async def _get_az_openai_embeddings(
        self, model_name: str, auth_context: AuthContext
    ) -> t.Optional[EmbeddingModel]:
        if model_name in self._AZURE_OPENAI_MODELS:
            embeddings = get_embeddings_model(
                api_key=auth_context.api_key,
                model_config=EmbeddingsModelConfig(
                    deployment=EmbeddingModelsEnum(model_name),
                ),
            )
            length = await self._get_embedding_model_length(model_name=model_name, model=embeddings)
            embedding_model = EmbeddingModel(
                name=model_name,
                model=embeddings,
                embedding_length=length,
                is_normalized_to_one=True,  # Azure OpenAI embeddings are normalized to 1
            )
            return embedding_model
        return None

    async def get(self, model_name: str, auth_context: AuthContext) -> EmbeddingModel:
        if model := await self._get_az_openai_embeddings(model_name, auth_context):
            return model
        raise ValueError(f"Unknown model name: {model_name}")

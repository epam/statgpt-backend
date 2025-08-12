import os

from pydantic import SecretStr

from common.config.utils import get_secret_env


class DialRagConfig:

    DIAL_RAG_DEPLOYMENT_ID = os.getenv("DIAL_RAG_DEPLOYMENT_ID", "dial-rag-pgvector")

    # set them up only to route requests to remote DIAL RAG
    DIAL_RAG_PGVECTOR_URL: str | None = os.getenv("DIAL_RAG_PGVECTOR_URL")
    DIAL_RAG_PGVECTOR_API_KEY: SecretStr | None = get_secret_env("DIAL_RAG_PGVECTOR_API_KEY")

    METADATA_ENDPOINT = "/indexing/documents/metadata"

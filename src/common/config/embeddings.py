import os


class EmbeddingsConfig:
    DEFAULT_MODEL = os.getenv("EMBEDDING_DEFAULT_MODEL", default="text-embedding-3-large")

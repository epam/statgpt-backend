import os


class VectorSearchConfig:
    INDICATOR_AGENT_TOP_K: int = int(os.getenv('INDICATOR_AGENT_TOP_K', 30))

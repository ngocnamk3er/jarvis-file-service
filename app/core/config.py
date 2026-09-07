from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Jarvis File Service"
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"

    DATABASE_URL: str = "postgresql://jarvis:jarvis@localhost:5433/jarvis_files"

    # Shared secret jarvis-backend sends on every call — this service is
    # internal-only (never reachable from the frontend/ingress), so a plain
    # header check is enough; it doesn't need its own JWKS/JWT verification.
    INTERNAL_API_KEY: str = ""

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "jarvis"
    MINIO_SECRET_KEY: str = "jarvis12345"
    MINIO_BUCKET: str = "jarvis-files"
    MINIO_SECURE: bool = False

    # Phase 3 — embeddings for search_files. Any OpenAI-compatible /embeddings
    # endpoint works; we point at OpenRouter (one key, many providers). Leave
    # EMBEDDING_API_KEY empty to run grep-only — ingest still extracts text,
    # search_vector just returns "not configured".
    #
    # Model: openai/text-embedding-3-large at 1024 dims (Matryoshka). Measured
    # on BEIR — SciFact nDCG@10 0.77, NFCorpus 0.41 — well above bge-m3's
    # 0.65 / 0.32. bge-m3 (multilingual, 1024-dim, $0.01/M) is the fallback if
    # Vietnamese retrieval ever needs it back.
    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://openrouter.ai/api/v1"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-large"
    # Requested output dimensionality. Only sent to the API when set — bge-m3
    # rejects the param; text-embedding-3 needs it for a non-native size.
    # MUST match the Qdrant collection (see app/search/vector_store.py).
    EMBEDDING_DIMENSIONS: int | None = 1024

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

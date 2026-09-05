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
    # endpoint works; we point at OpenRouter (one key, many providers) and use
    # BAAI's bge-m3 for its multilingual (incl. Vietnamese) retrieval quality
    # at $0.01/M tokens. Leave EMBEDDING_API_KEY empty to run grep-only —
    # ingest still extracts text, search_vector just returns "not configured".
    QDRANT_URL: str = "http://localhost:6333"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://openrouter.ai/api/v1"
    EMBEDDING_MODEL: str = "baai/bge-m3"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

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

    # Phase 3 (embeddings + search_files) — not wired up yet, see
    # ~/.claude/plans/cosmic-drifting-pinwheel.md.
    QDRANT_URL: str = "http://localhost:6333"
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

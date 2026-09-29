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

    # OCR for pages and images with no text layer (services/layout.py).
    # On by default now that the wheels are in requirements.txt. Still
    # checked at import as well as by this flag, so an image built without
    # them — or a local checkout that skipped the ~1.5GB install — keeps to
    # the text-layer path instead of failing files it cannot read.
    OCR_ENABLED: bool = True
    # PaddleOCR's Vietnamese model. Its "vi" recognises Latin script with
    # Vietnamese diacritics; "en" drops the diacritics, which for this
    # corpus loses the words rather than just the accents.
    OCR_LANG: str = "vi"
    # Recognitions below this are dropped. OCR reports confidence per line
    # and the low end is mostly page furniture and scan artefacts, which
    # cost embedding budget and pollute retrieval.
    OCR_MIN_CONFIDENCE: float = 0.5
    # oneDNN acceleration. Off by default because it is not portable: on
    # this host PaddleOCR 3.7 dies inside it with "ConvertPirAttribute2
    # RuntimeAttribute not support", and the same wheels work fine with it
    # disabled. Turn it on where it is known good — it is meaningfully
    # faster — but verify a real page first.
    OCR_USE_MKLDNN: bool = False

    # Cross-encoder reranking over vector-search candidates. Optional: with no
    # key, search falls back to plain embedding order rather than failing.
    #
    # OpenRouter serves this, so the existing key works — but note rerank
    # models do NOT appear in its /models listing and are rejected by
    # /chat/completions, which makes them easy to conclude don't exist. They
    # are reachable only at POST /api/v1/rerank.
    #
    # qwen3-reranker is multilingual, which matters: retrieval here is often a
    # Vietnamese question against English documents. Measured on exactly that
    # case, it scored the right passage 1.0 and everything else under 0.0001.
    # Jina (https://api.jina.ai/v1, jina-reranker-v2-base-multilingual) and
    # Cohere both speak the same request/response shape if this ever needs to
    # move.
    RERANK_API_KEY: str = ""
    RERANK_BASE_URL: str = "https://openrouter.ai/api/v1"
    RERANK_MODEL: str = "qwen/qwen3-reranker-8b"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

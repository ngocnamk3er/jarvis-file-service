"""Sync OpenAI SDK wrapper, pointed at any OpenAI-compatible /embeddings
endpoint (OpenRouter by default — see config.py). ingest.py/search.py wrap
these in asyncio.to_thread() to keep them off the event loop."""

from openai import OpenAI

from app.core.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.EMBEDDING_API_KEY, base_url=settings.EMBEDDING_BASE_URL)
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = _get_client().embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]

"""Sync OpenAI SDK wrapper — see ingest.py/search.py for the
asyncio.to_thread() calls that keep these off the event loop."""

from openai import OpenAI

from app.core.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = _get_client().embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in response.data]

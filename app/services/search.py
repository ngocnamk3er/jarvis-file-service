"""Search over a user's files — two ways in, one index (the search_files
agent tool's `exact` argument picks between them).

Lives outside the repository module because it touches the embedding API
and Qdrant rather than Postgres. Results carry the page and boxes the
chunk came from when the source format had them — see services/layout.py
for how those are derived and why they are stored as fractions."""

import asyncio

from app.core.config import settings
from app.search import vector_store
from app.services.embeddings import embed


async def search_vector(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    """Semantic search, hybridized with a keyword pass over the same
    points (see vector_store.search_hybrid) — a plain-vector result that
    also happens to contain the query's literal words ranks higher, and
    a chunk the embedding missed but the words didn't can surface too.
    Still just one EMBEDDING_API_KEY call; the keyword side costs no
    extra round trip to any external API, only Qdrant."""
    if not settings.EMBEDDING_API_KEY:
        raise ValueError("Semantic search is not configured (EMBEDDING_API_KEY missing)")
    vectors = await asyncio.to_thread(embed, [query])
    return await asyncio.to_thread(vector_store.search_hybrid, user_id, vectors[0], query, top_k)


async def search_fulltext(user_id: str, query: str, top_k: int = 5, phrase: bool = True) -> list[dict]:
    """Exact-match search — no embedding call, so no EMBEDDING_API_KEY
    dependency; works even when semantic search is unconfigured."""
    return await asyncio.to_thread(vector_store.search_fulltext, user_id, query, top_k, phrase)

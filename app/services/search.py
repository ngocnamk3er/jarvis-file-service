"""Semantic search over a user's files (the search_files agent tool).

Lives outside the repository module because it touches the embedding API
and Qdrant rather than Postgres. Results carry the page and boxes the
chunk came from when the source format had them — see services/layout.py
for how those are derived and why they are stored as fractions."""

import asyncio

from app.core.config import settings
from app.search import vector_store
from app.services.embeddings import embed


async def search_vector(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    if not settings.EMBEDDING_API_KEY:
        raise ValueError("Semantic search is not configured (EMBEDDING_API_KEY missing)")
    vectors = await asyncio.to_thread(embed, [query])
    return await asyncio.to_thread(vector_store.search, user_id, vectors[0], top_k)

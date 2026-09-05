"""Semantic half of hybrid search (search_files agent tool). The keyword
half (grep_files) lives in db/repository.py's search_grep — it only touches
Postgres, so it didn't need a home outside the repository module; this one
touches OpenAI + Qdrant instead, hence its own module."""

import asyncio

from app.core.config import settings
from app.search import vector_store
from app.services.embeddings import embed


async def search_vector(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    if not settings.EMBEDDING_API_KEY:
        raise ValueError("Semantic search is not configured (EMBEDDING_API_KEY missing)")
    vectors = await asyncio.to_thread(embed, [query])
    return await asyncio.to_thread(vector_store.search, user_id, vectors[0], top_k)

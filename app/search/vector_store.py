"""Thin wrapper around qdrant-client. One collection (`file_chunks`) shared
across every user — isolation is enforced by a `user_id` payload filter on
every query/delete, never by a separate collection per user."""

from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import settings

_COLLECTION = "file_chunks"
# Must match what embed() returns — currently openai/text-embedding-3-large
# truncated to 1024 dims (see EMBEDDING_DIMENSIONS in config.py). Changing
# the size means dropping + rebuilding the collection.
_VECTOR_SIZE = settings.EMBEDDING_DIMENSIONS or 1024

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Returns a client with the collection guaranteed to exist. The
    existence check runs every call (one cheap request) rather than once at
    first init — otherwise a collection dropped out from under a
    long-running process (Qdrant restart with lost volume, manual ops)
    would make every subsequent upsert/search fail until a redeploy."""
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL)
    if not _client.collection_exists(_COLLECTION):
        _client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=qm.VectorParams(size=_VECTOR_SIZE, distance=qm.Distance.COSINE),
        )
    return _client


def upsert_chunks(file_id: str, user_id: str, path: str, chunks: list[str], vectors: list[list[float]]) -> None:
    # A re-ingest (re-upload with the same file id would only happen via a
    # future "replace content" feature, not today's API — this is mainly a
    # safety net) must not leave stale chunks behind, so clear this file's
    # old points before inserting the new ones.
    delete_by_file_id(file_id)
    points = [
        qm.PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={"user_id": user_id, "file_id": file_id, "chunk_index": i, "chunk_text": chunk, "path": path},
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    if points:
        _get_client().upsert(collection_name=_COLLECTION, points=points)


def delete_by_file_id(file_id: str) -> None:
    _get_client().delete(
        collection_name=_COLLECTION,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(must=[qm.FieldCondition(key="file_id", match=qm.MatchValue(value=file_id))])
        ),
    )


def search(user_id: str, query_vector: list[float], top_k: int) -> list[dict]:
    # QdrantClient.search() was removed in qdrant-client 1.1x — query_points()
    # is the current API (verified live against 1.19.0).
    response = _get_client().query_points(
        collection_name=_COLLECTION,
        query=query_vector,
        query_filter=qm.Filter(must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]),
        limit=top_k,
    )
    return [
        {
            "file_id": h.payload["file_id"],
            "path": h.payload["path"],
            "chunk_text": h.payload["chunk_text"],
            "score": h.score,
        }
        for h in response.points
    ]

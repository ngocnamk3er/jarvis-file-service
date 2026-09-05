"""Post-upload pipeline, run via FastAPI BackgroundTasks (see
api/v1/endpoints/files.py's upload_file) so the upload request itself
returns as soon as the bytes are in MinIO, without waiting on extraction or
embedding.

1. Fetch bytes back from MinIO, extract plaintext, save it — that alone is
   enough to power grep_files.
2. If embeddings are configured (OPENAI_API_KEY set), chunk the extracted
   text, embed each chunk, and upsert into Qdrant — powers search_files.
   Left unconfigured, indexing stops at "done" after step 1: grep still
   works, semantic search just returns nothing for this file.
"""

import asyncio

from app.core.config import settings
from app.db import repository
from app.search import vector_store
from app.services.chunking import chunk_text
from app.services.embeddings import embed
from app.services.extraction import extract_text
from app.storage import minio_client


async def process_file(node_id: str) -> None:
    node = await repository.get_file_for_ingest(node_id)
    if node is None or node["type"] != "file" or not node["minio_object_key"]:
        return

    await repository.set_indexing_status(node_id, "extracting")
    try:
        content = await asyncio.to_thread(minio_client.get_object, node["minio_object_key"])
        text = extract_text(node["name"], node["mime_type"], content)
    except Exception:
        await repository.set_indexing_status(node_id, "failed")
        return

    if text is None:
        await repository.set_indexing_status(node_id, "skipped")
        return

    await repository.set_extracted_text(node_id, text)

    if not settings.OPENAI_API_KEY:
        await repository.set_indexing_status(node_id, "done")
        return

    await repository.set_indexing_status(node_id, "embedding")
    try:
        chunks = chunk_text(text)
        vectors = await asyncio.to_thread(embed, chunks)
        path = await repository.get_display_path(node_id)
        await asyncio.to_thread(vector_store.upsert_chunks, node_id, node["user_id"], path, chunks, vectors)
    except Exception:
        await repository.set_indexing_status(node_id, "failed")
        return

    await repository.set_indexing_status(node_id, "done")

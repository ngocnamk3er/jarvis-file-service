"""Post-upload pipeline, run via FastAPI BackgroundTasks (see
api/v1/endpoints/files.py's upload_file) so the upload request itself
returns as soon as the bytes are in MinIO, without waiting on extraction or
embedding.

1. Fetch bytes back from MinIO and read it. A PDF or an image goes through
   layout.extract_spans first, which keeps the page and box each piece of
   text came from; anything else (or a PDF that yields nothing) falls back
   to flat extraction. Either way a plaintext rendering is stored, so
   read_file behaves the same for every format.
2. If embeddings are configured (EMBEDDING_API_KEY set), chunk, embed and
   upsert into Qdrant — powers search_files. Chunks built from spans carry
   page/boxes/offsets into Qdrant's payload so a hit can be pointed at on
   the page; chunks built from flat text carry none, and a caller reads
   their absence as "this format has no coordinates" rather than as a
   missing feature.

   Left unconfigured, indexing stops at "done" after step 1: semantic
   search just returns nothing for this file.
"""

import asyncio

from app.core.config import settings
from app.db import repository
from app.search import vector_store
from app.services.chunking import Chunk, chunk_spans, chunk_text, spans_to_text
from app.services.embeddings import embed
from app.services.extraction import extract_text
from app.services.layout import extract_spans
from app.storage import minio_client


async def process_file(node_id: str) -> None:
    node = await repository.get_file_for_ingest(node_id)
    if node is None or node["type"] != "file" or not node["minio_object_key"]:
        return

    await repository.set_indexing_status(node_id, "extracting")
    try:
        content = await asyncio.to_thread(minio_client.get_object, node["minio_object_key"])
        # Positioned first, flat second. OCR and PDF layout parsing are
        # CPU-bound and slow enough to block the event loop for seconds on
        # a long document, hence the thread.
        spans = await asyncio.to_thread(
            extract_spans, node["name"], node["mime_type"], content
        )
        text = spans_to_text(spans) if spans else extract_text(
            node["name"], node["mime_type"], content
        )
    except Exception:
        await repository.set_indexing_status(node_id, "failed")
        return

    if text is None:
        await repository.set_indexing_status(node_id, "skipped")
        return

    await repository.set_extracted_text(node_id, text)

    if not settings.EMBEDDING_API_KEY:
        await repository.set_indexing_status(node_id, "done")
        return

    await repository.set_indexing_status(node_id, "embedding")
    try:
        chunks = (
            chunk_spans(spans)
            if spans
            else [Chunk(text=t, page_start=0, page_end=0) for t in chunk_text(text)]
        )
        vectors = await asyncio.to_thread(embed, [c.text for c in chunks])
        path = await repository.get_display_path(node_id)
        await asyncio.to_thread(
            vector_store.upsert_chunks, node_id, node["user_id"], path, chunks, vectors
        )
    except Exception:
        await repository.set_indexing_status(node_id, "failed")
        return

    await repository.set_indexing_status(node_id, "done")

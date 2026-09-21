import asyncio

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Response, UploadFile

from app.api.deps import verify_internal_key
from app.db import repository
from app.schemas.files import (
    RerankRequest,
    RerankResult,
    FileNodeOut,
    FileSearchResult,
    FolderCreate,
    NodeDetail,
    NodeUpdate,
    VectorSearchRequest,
    VectorSearchResult,
)
from app.services import ingest, rerank as rerank_service, search as search_service
from app.storage import minio_client

router = APIRouter(dependencies=[Depends(verify_internal_key)])


@router.post("/folders", response_model=FileNodeOut)
async def create_folder(body: FolderCreate):
    try:
        return await repository.create_folder(body.user_id, body.parent_path, body.name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/tree", response_model=list[FileNodeOut])
async def list_tree(user_id: str, path: str = "/"):
    try:
        return await repository.list_children(user_id, path)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/files", response_model=FileNodeOut)
async def upload_file(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    parent_path: str = Form("/"),
    file: UploadFile = File(...),
):
    content = await file.read()
    try:
        node = await repository.create_file(user_id, parent_path, file.filename, content, file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    # Returns immediately ("pending"); extraction runs after the response is
    # sent — see services/ingest.py.
    background_tasks.add_task(ingest.process_file, node["id"])
    return node


@router.get("/nodes/{node_id}", response_model=NodeDetail | None)
async def get_node(node_id: str, user_id: str):
    return await repository.get_node(user_id, node_id)


@router.get("/nodes/{node_id}/content")
async def get_node_content(node_id: str, user_id: str):
    """Raw bytes, for preview (inline) and download in the frontend — the
    frontend fetches this with the browser's Authorization header attached
    and turns it into a blob URL itself; the Content-Disposition here is
    just a filename hint, not what decides preview vs. download (that's a
    frontend-side choice: <img>/<iframe> the blob vs. <a download>)."""
    ref = await repository.get_content_ref(user_id, node_id)
    if ref is None or not ref["minio_object_key"]:
        raise HTTPException(status_code=404, detail="Not found")
    content = await asyncio.to_thread(minio_client.get_object, ref["minio_object_key"])
    return Response(
        content=content,
        media_type=ref["mime_type"] or "application/octet-stream",
        headers={"Content-Disposition": f'inline; filename="{ref["name"]}"'},
    )


@router.get("/resolve", response_model=NodeDetail | None)
async def resolve_path(user_id: str, path: str):
    """Turns a "/a/b.txt"-style path into a node — used by the read_file/
    grep_files agent tools, which only have a path string. GET /tree already
    covers the list_files tool directly (it takes a path natively)."""
    return await repository.resolve_path(user_id, path)


@router.patch("/nodes/{node_id}", response_model=FileNodeOut)
async def update_node(node_id: str, user_id: str, body: NodeUpdate):
    try:
        return await repository.rename_or_move(user_id, node_id, body.name, body.parent_path)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str, user_id: str):
    await repository.delete_node(user_id, node_id)
    return {"ok": True}


@router.get("/search/grep", response_model=list[FileSearchResult])
async def grep_files(user_id: str, query: str, path: str = "/"):
    """Plain keyword search over name + extracted content — the
    grep_files agent tool. See POST /search/vector (phase 3) for the
    semantic half of hybrid search."""
    try:
        return await repository.search_grep(user_id, query, path)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/reindex")
async def reindex(user_id: str | None = None):
    """Re-run the ingest pipeline (extract → embed → Qdrant) over existing
    files — for after an embedding-model change or to recover failed rows.
    Internal/admin only; synchronous since this deployment's file count is
    small. Omit user_id to reindex everyone's."""
    ids = await repository.list_file_ids(user_id)
    for node_id in ids:
        await ingest.process_file(node_id)
    return {"reindexed": len(ids)}


@router.post("/search/vector", response_model=list[VectorSearchResult])
async def search_vector(body: VectorSearchRequest):
    """Semantic search over embedded chunks — the search_files agent tool.
    See GET /search/grep for the keyword half of hybrid search."""
    try:
        return await search_service.search_vector(body.user_id, body.query, body.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/search/rerank", response_model=list[RerankResult])
async def rerank_documents(body: RerankRequest):
    """Reorder candidate passages by how well they answer `query`.

    Separate from /search/vector because the caller assembles the pool first:
    agentic retrieval issues several sub-queries and merges their results, and
    reranking is only meaningful once that merge has happened.

    With no reranker configured this returns the input order with zero scores,
    so the caller can treat "no opinion" and "unconfigured" the same way.
    """
    pairs = await rerank_service.rerank(body.query, body.documents, body.top_n)
    return [RerankResult(index=i, score=s) for i, s in pairs]

from datetime import datetime
from pydantic import BaseModel


class FileNodeOut(BaseModel):
    id: str
    user_id: str
    parent_id: str | None
    name: str
    type: str
    mime_type: str | None = None
    size_bytes: int | None = None
    indexing_status: str
    created_at: datetime
    updated_at: datetime


class NodeDetail(FileNodeOut):
    """GET /nodes/{id} and GET /resolve — adds extracted_text, which
    list/tree responses deliberately omit to keep listings light."""

    extracted_text: str | None = None


class FileSearchResult(FileNodeOut):
    """GET /search/grep — adds the display path, since a search result's
    main value over a bare tree listing is telling the caller where it is."""

    path: str


class FolderCreate(BaseModel):
    user_id: str
    parent_path: str = "/"
    name: str


class NodeUpdate(BaseModel):
    name: str | None = None
    parent_path: str | None = None


class VectorSearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 5


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_n: int | None = None


class RerankResult(BaseModel):
    """`index` refers back into the caller's `documents` list — the caller
    holds the full chunk metadata, so only the ordering travels back."""

    index: int
    score: float


class VectorSearchResult(BaseModel):
    file_id: str
    path: str
    chunk_text: str
    score: float

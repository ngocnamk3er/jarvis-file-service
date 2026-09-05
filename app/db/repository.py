import asyncio
from uuid import uuid4

from sqlalchemy import select, delete, or_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.connection import get_sessionmaker
from app.db.models import FileNode
from app.storage import minio_client


def _node_dict(n: FileNode) -> dict:
    return {
        "id": n.id,
        "user_id": n.user_id,
        "parent_id": n.parent_id,
        "name": n.name,
        "type": n.type,
        "mime_type": n.mime_type,
        "size_bytes": n.size_bytes,
        "indexing_status": n.indexing_status,
        "created_at": n.created_at,
        "updated_at": n.updated_at,
    }


def _split_path(path: str) -> list[str]:
    return [p for p in path.strip("/").split("/") if p]


async def _resolve_folder(session: AsyncSession, user_id: str, path: str) -> FileNode | None:
    """Resolve a "/a/b/c" path to its FileNode, walking one folder segment at
    a time from the root. Returns None for the root itself ("/" or "") — it
    has no row of its own, since parent_id=None already means "root" for
    every top-level node. Also returns None if any segment doesn't exist or
    isn't a folder; callers distinguish "root" from "not found" by checking
    whether `path` was non-empty."""
    parts = _split_path(path)
    parent_id: str | None = None
    node: FileNode | None = None
    for part in parts:
        result = await session.execute(
            select(FileNode).where(
                FileNode.user_id == user_id,
                # SQLAlchemy's `==` special-cases None into "IS NULL", so this
                # works for both a root-level lookup (parent_id=None) and a
                # nested one.
                FileNode.parent_id == parent_id,
                FileNode.name == part,
                FileNode.type == "folder",
            )
        )
        node = result.scalar_one_or_none()
        if node is None:
            return None
        parent_id = node.id
    return node


async def _resolve_node(session: AsyncSession, user_id: str, path: str) -> FileNode | None:
    """Like _resolve_folder, but the last path segment may be a file too —
    used by the read_file/grep_files agent tools' `path` argument."""
    parts = _split_path(path)
    if not parts:
        return None
    *parent_parts, last = parts
    parent_id: str | None = None
    if parent_parts:
        parent = await _resolve_folder(session, user_id, "/".join(parent_parts))
        if parent is None:
            return None
        parent_id = parent.id
    result = await session.execute(
        select(FileNode).where(
            FileNode.user_id == user_id,
            FileNode.parent_id == parent_id,
            FileNode.name == last,
        )
    )
    return result.scalar_one_or_none()


async def _check_name_free(
    session: AsyncSession, user_id: str, parent_id: str | None, name: str, exclude_id: str | None = None
) -> None:
    stmt = select(FileNode).where(
        FileNode.user_id == user_id,
        FileNode.parent_id == parent_id,
        FileNode.name == name,
    )
    if exclude_id is not None:
        stmt = stmt.where(FileNode.id != exclude_id)
    result = await session.execute(stmt)
    if result.scalar_one_or_none() is not None:
        raise ValueError(f"'{name}' already exists in that folder")


async def _resolve_parent_or_raise(session: AsyncSession, user_id: str, parent_path: str) -> str | None:
    if not _split_path(parent_path):
        return None
    parent = await _resolve_folder(session, user_id, parent_path)
    if parent is None:
        raise ValueError(f"Parent folder not found: {parent_path}")
    return parent.id


async def create_folder(user_id: str, parent_path: str, name: str) -> dict:
    async with get_sessionmaker()() as session:
        parent_id = await _resolve_parent_or_raise(session, user_id, parent_path)
        await _check_name_free(session, user_id, parent_id, name)

        node = FileNode(id=str(uuid4()), user_id=user_id, parent_id=parent_id, name=name, type="folder")
        session.add(node)
        await session.commit()
        await session.refresh(node)
        return _node_dict(node)


async def list_children(user_id: str, path: str) -> list[dict]:
    async with get_sessionmaker()() as session:
        parent_id = await _resolve_parent_or_raise(session, user_id, path)
        result = await session.execute(
            select(FileNode)
            .where(FileNode.user_id == user_id, FileNode.parent_id == parent_id)
            .order_by(FileNode.type.desc(), FileNode.name)  # folders first, then alphabetical
        )
        return [_node_dict(n) for n in result.scalars()]


async def create_file(
    user_id: str, parent_path: str, name: str, content: bytes, mime_type: str | None
) -> dict:
    async with get_sessionmaker()() as session:
        parent_id = await _resolve_parent_or_raise(session, user_id, parent_path)
        await _check_name_free(session, user_id, parent_id, name)

        node_id = str(uuid4())
        object_key = f"{user_id}/{node_id}"
        # Blocking SDK call — off the event loop (see minio_client's docstring).
        await asyncio.to_thread(minio_client.put_object, object_key, content, mime_type)

        node = FileNode(
            id=node_id,
            user_id=user_id,
            parent_id=parent_id,
            name=name,
            type="file",
            mime_type=mime_type,
            size_bytes=len(content),
            minio_object_key=object_key,
        )
        session.add(node)
        await session.commit()
        await session.refresh(node)
        return _node_dict(node)


async def get_file_for_ingest(node_id: str) -> dict | None:
    """Internal-only accessor for services/ingest.py — unlike get_node(),
    this exposes minio_object_key (never part of the public API shape) and
    skips the user_id filter since the caller (upload_file's own background
    task) already knows it created this exact row."""
    async with get_sessionmaker()() as session:
        result = await session.execute(select(FileNode).where(FileNode.id == node_id))
        node = result.scalar_one_or_none()
        if node is None:
            return None
        return {
            "name": node.name,
            "type": node.type,
            "mime_type": node.mime_type,
            "minio_object_key": node.minio_object_key,
            "user_id": node.user_id,
        }


async def get_display_path(node_id: str) -> str:
    """Walks parent_id up to the root, for the Qdrant payload's `path` field
    (so search_files results can cite a human-readable location)."""
    async with get_sessionmaker()() as session:
        parts: list[str] = []
        current_id: str | None = node_id
        while current_id is not None:
            result = await session.execute(select(FileNode).where(FileNode.id == current_id))
            node = result.scalar_one_or_none()
            if node is None:
                break
            parts.append(node.name)
            current_id = node.parent_id
        return "/" + "/".join(reversed(parts))


async def get_node(user_id: str, node_id: str) -> dict | None:
    async with get_sessionmaker()() as session:
        result = await session.execute(
            select(FileNode).where(FileNode.id == node_id, FileNode.user_id == user_id)
        )
        node = result.scalar_one_or_none()
        if node is None:
            return None
        d = _node_dict(node)
        d["extracted_text"] = node.extracted_text
        return d


async def resolve_path(user_id: str, path: str) -> dict | None:
    """Turns a "/a/b.txt"-style path into a node — what read_file/grep_files
    (agent tools) use, since they only have a path string, not an id."""
    async with get_sessionmaker()() as session:
        node = await _resolve_node(session, user_id, path)
        if node is None:
            return None
        d = _node_dict(node)
        d["extracted_text"] = node.extracted_text
        return d


async def rename_or_move(user_id: str, node_id: str, name: str | None, parent_path: str | None) -> dict:
    async with get_sessionmaker()() as session:
        result = await session.execute(
            select(FileNode).where(FileNode.id == node_id, FileNode.user_id == user_id)
        )
        node = result.scalar_one_or_none()
        if node is None:
            raise ValueError("Not found")

        new_parent_id = node.parent_id
        if parent_path is not None:
            new_parent_id = await _resolve_parent_or_raise(session, user_id, parent_path)
        new_name = name if name is not None else node.name

        await _check_name_free(session, user_id, new_parent_id, new_name, exclude_id=node_id)

        node.parent_id = new_parent_id
        node.name = new_name
        await session.commit()
        await session.refresh(node)
        return _node_dict(node)


async def _collect_descendants(session: AsyncSession, node_id: str) -> list[FileNode]:
    """BFS over parent_id — every node (folder or file) under node_id, not
    including node_id itself. Shared by delete_node (needs MinIO object keys
    before the Postgres cascade fires — ON DELETE CASCADE knows nothing about
    MinIO) and search_grep (needs the id set to scope a subtree search)."""
    all_nodes: list[FileNode] = []
    frontier = [node_id]
    while frontier:
        result = await session.execute(select(FileNode).where(FileNode.parent_id.in_(frontier)))
        children = list(result.scalars())
        frontier = [c.id for c in children]
        all_nodes.extend(children)
    return all_nodes


async def delete_node(user_id: str, node_id: str) -> None:
    async with get_sessionmaker()() as session:
        result = await session.execute(
            select(FileNode).where(FileNode.id == node_id, FileNode.user_id == user_id)
        )
        node = result.scalar_one_or_none()
        if node is None:
            return

        descendants = await _collect_descendants(session, node_id)
        object_keys = [d.minio_object_key for d in descendants if d.minio_object_key]
        file_ids = [d.id for d in descendants if d.type == "file"]
        if node.minio_object_key:
            object_keys.append(node.minio_object_key)
        if node.type == "file":
            file_ids.append(node.id)

        await session.execute(delete(FileNode).where(FileNode.id == node_id))
        await session.commit()

    for key in object_keys:
        await asyncio.to_thread(minio_client.delete_object, key)

    if settings.OPENAI_API_KEY:
        # Local import: keeps a hard Qdrant dependency out of this function
        # for phase-1/2-only deployments — without OPENAI_API_KEY nothing
        # was ever embedded, so there's nothing in Qdrant to clean up.
        from app.search import vector_store

        for file_id in file_ids:
            await asyncio.to_thread(vector_store.delete_by_file_id, file_id)


async def search_grep(user_id: str, query: str, path: str = "/", limit: int = 50) -> list[dict]:
    """ILIKE over name + extracted_text, optionally scoped to the subtree
    rooted at `path`. No Qdrant/embeddings involved — this is the plain
    keyword-search half of hybrid search; see search_vector (phase 3) for
    the semantic half."""
    async with get_sessionmaker()() as session:
        scope_ids: list[str] | None = None
        if _split_path(path):
            node = await _resolve_node(session, user_id, path)
            if node is None:
                raise ValueError(f"Path not found: {path}")
            if node.type == "file":
                scope_ids = [node.id]
            else:
                scope_ids = [node.id] + [d.id for d in await _collect_descendants(session, node.id)]

        stmt = select(FileNode).where(
            FileNode.user_id == user_id,
            FileNode.type == "file",
            or_(FileNode.name.ilike(f"%{query}%"), FileNode.extracted_text.ilike(f"%{query}%")),
        )
        if scope_ids is not None:
            stmt = stmt.where(FileNode.id.in_(scope_ids))
        result = await session.execute(stmt.order_by(FileNode.name).limit(limit))
        matches = list(result.scalars())

    # Display path per match — grep_files' main value over a bare listing is
    # telling the caller *where* a hit lives. Result sets are capped at
    # `limit`, so the extra per-match round trip stays cheap.
    out = []
    for n in matches:
        d = _node_dict(n)
        d["path"] = await get_display_path(n.id)
        out.append(d)
    return out


async def set_indexing_status(node_id: str, status: str) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(update(FileNode).where(FileNode.id == node_id).values(indexing_status=status))
        await session.commit()


async def set_extracted_text(node_id: str, text: str) -> None:
    async with get_sessionmaker()() as session:
        await session.execute(update(FileNode).where(FileNode.id == node_id).values(extracted_text=text))
        await session.commit()

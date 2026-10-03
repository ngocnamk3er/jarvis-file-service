"""Thin wrapper around qdrant-client. One collection (`file_chunks`) shared
across every user — isolation is enforced by a `user_id` payload filter on
every query/delete, never by a separate collection per user."""

import unicodedata
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

# Full-text index on the same field semantic search already reads —
# keyword search is a second way to query the exact points vector search
# uses, not a second store, so every hit carries the same page/boxes/char
# offsets with no extra plumbing. Settings here matter and were each
# verified live before being hard-coded:
#   ascii_folding  catches the common case of a Vietnamese query typed
#                  without diacritics (e.g. "nghi phep") against text
#                  stored with them ("nghỉ phép") — confirmed live, off by
#                  default.
#   phrase_matching lets MatchPhrase require the query's words adjacent
#                  and in order; default is False, and without it
#                  MatchPhrase silently returns zero hits for every
#                  query — not an error, just nothing, which is the kind
#                  of failure that looks like "no results" rather than
#                  "misconfigured".
_CHUNK_TEXT_INDEX = qm.TextIndexParams(
    type=qm.TextIndexType.TEXT,
    tokenizer=qm.TokenizerType.WORD,
    min_token_len=2,
    lowercase=True,
    ascii_folding=True,
    phrase_matching=True,
)


def _get_client() -> QdrantClient:
    """Returns a client with the collection and its full-text index
    guaranteed to exist. Both checks run every call (cheap requests)
    rather than once at first init — otherwise a collection dropped out
    from under a long-running process (Qdrant restart with lost volume,
    manual ops) would make every subsequent upsert/search fail until a
    redeploy, and the same for an index dropped or never created because
    this code shipped after the collection already existed."""
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.QDRANT_URL)
    if not _client.collection_exists(_COLLECTION):
        _client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=qm.VectorParams(size=_VECTOR_SIZE, distance=qm.Distance.COSINE),
        )
    if "chunk_text" not in _client.get_collection(_COLLECTION).payload_schema:
        _client.create_payload_index(
            _COLLECTION, field_name="chunk_text", field_schema=_CHUNK_TEXT_INDEX
        )
    return _client


def upsert_chunks(file_id: str, user_id: str, path: str, chunks, vectors: list[list[float]]) -> None:
    """`chunks` are services.chunking.Chunk. Their page/boxes/offsets ride
    along in the payload, which Qdrant stores schemalessly — so adding them
    needed no collection rebuild, and points written before they existed
    stay readable, just without the new keys.

    A chunk from a format with no coordinates carries page 0 and no boxes.
    Zero rather than null because it stays comparable in a Qdrant filter,
    and because page numbers are 1-based everywhere else, so 0 cannot
    collide with a real page."""
    # A re-ingest (re-upload with the same file id would only happen via a
    # future "replace content" feature, not today's API — this is mainly a
    # safety net) must not leave stale chunks behind, so clear this file's
    # old points before inserting the new ones.
    delete_by_file_id(file_id)
    points = [
        qm.PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload={
                "user_id": user_id,
                "file_id": file_id,
                "chunk_index": i,
                "chunk_text": chunk.text,
                "path": path,
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                # [[page, [x0, top, x1, bottom], [local_start, local_end]], ...]
                # — fractions of the page, origin top-left; the char range
                # is local to chunk.text, used to narrow boxes to a match
                # (see _narrow_boxes). Lists, not tuples, so they round-trip
                # through Qdrant's JSON unchanged.
                "boxes": [
                    [page, list(box), list(char_range)] for page, box, char_range in chunk.boxes
                ],
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            },
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


def _strip_box_offsets(boxes: list) -> list:
    """[page, bbox, (local_start, local_end)] -> [page, bbox] — the char
    range is this module's own detail for narrowing a chunk's boxes to a
    match (see _narrow_boxes); VectorSearchResult only ever declares
    page + bbox, both for points indexed before per-box char ranges
    existed (2-element boxes — box[:2] is a no-op on those) and because
    no caller outside this module has a use for the range itself."""
    return [box[:2] for box in boxes]


def _hit(payload: dict, score: float) -> dict:
    """One search result, whichever of the two query paths below found it.

    .get, not [], on the citation fields: points upserted before they
    existed have no such keys, and a stale point should degrade to "no
    coordinates" rather than break the whole search.
    """
    return {
        "file_id": payload["file_id"],
        "path": payload["path"],
        "chunk_text": payload["chunk_text"],
        "score": score,
        "page_start": payload.get("page_start", 0),
        "page_end": payload.get("page_end", 0),
        "boxes": _strip_box_offsets(payload.get("boxes", [])),
        "char_start": payload.get("char_start", 0),
        "char_end": payload.get("char_end", 0),
    }


def _fold(s: str) -> str:
    """Strip combining diacritics, one base character left per input
    character — NFD-decomposing a precomposed accented letter (Vietnamese
    text is typically one codepoint per letter, e.g. "ệ") splits it into
    exactly one base letter plus its combining marks, so dropping marks
    of category Mn leaves a result the same *length* as the input. That
    1:1 positional mapping is the only reason this is safe to use for
    locating a match: a position found in folded text is the same
    position in the original, unfolded one.
    """
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _find_match(text: str, query: str) -> tuple[int, int] | None:
    """Where `query` actually sits in `text`, diacritic- and
    case-insensitive.

    Folding both sides before comparing matters, not just for a bare
    "nghi phep" query to find text stored as "nghỉ phép" at all (what
    Qdrant's ascii_folding index already does for the search itself) —
    without it, this fell back word-by-word on an unfolded miss and
    matched "nghi" as a bare substring *inside* an unrelated word,
    "nghiệp" (unemployment), while hunting for "nghỉ phép" (leave) —
    confirmed live 2026-10-03, narrowing a citation to the wrong line
    entirely. Folding first lets the whole phrase match directly where
    it belongs, so that wrong-word fallback never triggers.

    Tries the whole phrase first, then the first individual word — a
    non-phrase (any-order) match can have its words scattered wider than
    one contiguous span covers. Returns None, not a guess, when nothing
    folds into a match anywhere.
    """
    folded_text, folded_query = _fold(text).lower(), _fold(query).lower()
    idx = folded_text.find(folded_query)
    if idx != -1:
        return idx, idx + len(folded_query)
    for word in folded_query.split():
        idx = folded_text.find(word)
        if idx != -1:
            return idx, idx + len(word)
    return None


def _narrow_boxes(boxes: list, chunk_text: str, query: str) -> list:
    """`boxes` cut down to just the line(s) that actually contain where
    `query` matched, instead of every line in the chunk.

    A chunk can be the entire text of a short document — a 2-page PDF
    under the 300-word chunk threshold collapses to one chunk, whose
    boxes then cover the whole document (confirmed live 2026-10-03: a
    test file highlighted almost in full for an exact-match hit near
    its end). Exact search knows precisely which substring matched,
    unlike semantic search's whole-chunk relevance, so it can do better.

    Falls back to every box, offsets stripped, when there is nothing to
    narrow by: no verbatim match, or a box that predates the per-box
    char range this needs (2-element — indexed before this existed).
    """
    if not boxes or len(boxes[0]) < 3:
        return _strip_box_offsets(boxes)
    match = _find_match(chunk_text, query)
    if match is None:
        return _strip_box_offsets(boxes)
    start, end = match
    narrowed = [[page, bbox] for page, bbox, (cs, ce) in boxes if cs < end and ce > start]
    return narrowed or _strip_box_offsets(boxes)


def search(user_id: str, query_vector: list[float], top_k: int) -> list[dict]:
    # QdrantClient.search() was removed in qdrant-client 1.1x — query_points()
    # is the current API (verified live against 1.19.0).
    response = _get_client().query_points(
        collection_name=_COLLECTION,
        query=query_vector,
        query_filter=qm.Filter(must=[qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id))]),
        limit=top_k,
    )
    return [_hit(h.payload, h.score) for h in response.points]


def search_fulltext(user_id: str, query: str, top_k: int, phrase: bool = True) -> list[dict]:
    """Keyword search over the same chunk_text field semantic search
    queries, against the same points — so a hit here carries the same
    page/boxes/char offsets, no separate index to keep in sync.

    phrase=True (default) requires the query's words adjacent and in
    that order (MatchPhrase) — right for an exact code, a name, a quoted
    sentence. phrase=False only requires every word present somewhere in
    the chunk, any order (MatchText) — wider net, for when the caller
    has the right words but not the right order or isn't sure they're
    adjacent.

    No relevance score of its own: a phrase/text condition is a boolean
    filter, not a ranked query. Every hit gets score=1.0 rather than a
    number that would imply degrees of match that were never computed —
    order here is Qdrant's scroll order (insertion order), not
    relevance.
    """
    condition = qm.MatchPhrase(phrase=query) if phrase else qm.MatchText(text=query)
    points, _ = _get_client().scroll(
        collection_name=_COLLECTION,
        scroll_filter=qm.Filter(
            must=[
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                qm.FieldCondition(key="chunk_text", match=condition),
            ]
        ),
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    hits = []
    for p in points:
        hit = _hit(p.payload, 1.0)
        hit["boxes"] = _narrow_boxes(p.payload.get("boxes", []), p.payload["chunk_text"], query)
        hits.append(hit)
    return hits

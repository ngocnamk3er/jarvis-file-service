"""Cross-encoder reranking over candidates a vector search already found.

Embedding search answers "is this text about the same thing as the query" —
each side is encoded independently, so the comparison is between two summaries
of meaning. A cross-encoder reads the query and the passage *together* and
answers the question that actually matters: does this passage contain the
answer. That is why reranking a generous candidate pool beats simply taking
fewer nearest neighbours.

Hosted rather than local on purpose. Running `bge-reranker-v2-m3` in-process
would mean torch, transformers and a ~2GB download in a service whose entire
dependency list is currently fourteen light packages, on a cluster already
tight on memory. Both Jina and Cohere speak close enough to the same shape
that one client covers either — point RERANK_BASE_URL at whichever.

Optional throughout. With no key configured, `rerank()` returns the input
order untouched, so search degrades to plain vector similarity rather than
failing.
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 20.0
# Passages are truncated before being sent: rerankers cap input length, and
# the decisive sentence in a ~300-word chunk is almost never past this point.
_MAX_DOC_CHARS = 2_000


def configured() -> bool:
    return bool(settings.RERANK_API_KEY)


async def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """Score `documents` against `query`.

    Returns (original index, relevance score) ordered best first. Falls back to
    the input order with zero scores when reranking is unconfigured or fails —
    callers treat that as "no opinion", not as "all irrelevant".
    """
    if not documents:
        return []
    if not configured():
        return [(i, 0.0) for i in range(len(documents))]

    payload = {
        "model": settings.RERANK_MODEL,
        "query": query,
        "documents": [d[:_MAX_DOC_CHARS] for d in documents],
        "top_n": top_n or len(documents),
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{settings.RERANK_BASE_URL.rstrip('/')}/rerank",
                headers={"Authorization": f"Bearer {settings.RERANK_API_KEY}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.warning("rerank failed, keeping vector order", exc_info=True)
        return [(i, 0.0) for i in range(len(documents))]

    # Jina and Cohere both return {"results": [{"index": .., "relevance_score": ..}]}.
    out: list[tuple[int, float]] = []
    for item in data.get("results", []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(documents):
            out.append((idx, float(item.get("relevance_score", 0.0))))
    if not out:
        logger.warning("rerank returned no usable results, keeping vector order")
        return [(i, 0.0) for i in range(len(documents))]
    return out

"""Retrieval: find the chunks that best answer a question.

Pipeline: embed the question -> vector search and keyword search (run in
parallel) -> merge with Reciprocal Rank Fusion -> rerank with a cross-encoder ->
keep the best few. Each stage can be switched off through ``RetrievalConfig`` so
different configurations can be compared in the evaluation.

Every query is restricted to one user's chunks (and optionally a chosen set of
documents), so another user's content can never be returned.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, replace
from typing import Any

import voyageai

from app.config import settings
from app.db import connection
from app.ingestion.embed import embed_query
from app.ingestion.store import to_vector_literal

logger = logging.getLogger(__name__)

RERANK_TIMEOUT_SECONDS = 30
RERANK_MAX_RETRIES = 3
HNSW_EF_SEARCH = 200  # how many index entries pgvector examines; must be >= the LIMIT
MAX_KEYWORDS = 20

# Common English words carry no signal for keyword search (the index uses the
# language-neutral 'simple' configuration, which does not remove them itself).
_STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its just me more most my
    no nor not of off on once only or other our out over own same she should so some such than
    that the their them then there these they this those through to too under until up very was
    we were what when where which while who whom why will with would you your""".split()
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)


class RerankServiceError(Exception):
    """The reranking API failed or timed out (the caller can fall back to fused order)."""


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with the scores that explain why.

    Attributes:
        similarity: Cosine similarity between the question and the chunk (0 to 1
            for typical text). Computed for every candidate, including ones found
            only by keyword search.
        fused_score: Reciprocal Rank Fusion score from merging both searches.
        rerank_score: Relevance score from the reranker, or ``None`` if not reranked.
    """

    chunk_id: int
    document_id: int
    filename: str
    heading: str | None
    page_start: int | None
    page_end: int | None
    chunk_index: int
    text: str
    similarity: float
    fused_score: float = 0.0
    rerank_score: float | None = None

    @property
    def rerank_text(self) -> str:
        """The text the reranker reads: heading path plus content."""
        return f"{self.heading}\n\n{self.text}" if self.heading else self.text

    @property
    def location(self) -> str:
        """Where the chunk sits in its document, for citations."""
        if self.page_start is None:
            return self.heading or "document"
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


@dataclass(frozen=True)
class RetrievalConfig:
    """Switches and sizes for one retrieval configuration."""

    use_keyword: bool = True
    use_reranker: bool = True
    vector_k: int = settings.vector_candidates
    keyword_k: int = settings.keyword_candidates
    rerank_k: int = settings.rerank_candidates
    final_k: int = settings.final_top_k
    rrf_k: int = settings.rrf_k
    min_similarity: float = settings.min_similarity
    min_rerank_score: float = settings.min_rerank_score


@dataclass(frozen=True)
class RetrievalResult:
    """Outcome of one retrieval.

    Attributes:
        chunks: The best chunks, most relevant first.
        candidates_considered: How many distinct chunks the searches found.
        reranked: ``True`` if the reranker ordered the chunks (``False`` if it was
            switched off or failed).
        relevant: ``False`` when even the best chunk scores below the threshold,
            meaning the documents most likely do not answer the question.
        timings_ms: Milliseconds spent in each stage.
    """

    chunks: list[RetrievedChunk]
    candidates_considered: int
    reranked: bool
    relevant: bool
    timings_ms: dict[str, float]


def keyword_query(question: str) -> str | None:
    """Turn a question into a safe OR-query for PostgreSQL full-text search.

    Only letters, digits and underscores survive, so no search syntax can be
    injected. Returns ``None`` if nothing useful is left.
    """
    words: list[str] = []
    for word in _WORD_RE.findall(question.lower()):
        too_short = len(word) < 3 and not any(ch.isdigit() for ch in word)
        if word in _STOPWORDS or too_short or word in words:
            continue
        words.append(word)
    return " | ".join(words[:MAX_KEYWORDS]) or None


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: merge several ranked id lists into one score each.

    A chunk scores ``1 / (k + rank)`` in every list it appears in, so chunks
    found by both searches rise to the top, and no raw scores need calibrating.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for position, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
    return scores


_COLUMNS = """
    c.id, c.document_id, d.filename, c.heading, c.page_start, c.page_end,
    c.chunk_index, c.content, 1 - (c.embedding <=> %(vec)s::vector) AS similarity
"""


def _vector_search(
    vector: str, user_id: int, document_ids: list[int] | None, limit: int
) -> list[RetrievedChunk]:
    """Nearest chunks by cosine distance (uses the HNSW index)."""
    doc_filter = "AND c.document_id = ANY(%(docs)s)" if document_ids is not None else ""
    sql = f"""
        SELECT {_COLUMNS}
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.user_id = %(user)s {doc_filter}
        ORDER BY c.embedding <=> %(vec)s::vector
        LIMIT %(limit)s
    """
    params = {"vec": vector, "user": user_id, "docs": document_ids, "limit": limit}
    with connection() as conn:
        conn.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
        rows = conn.execute(sql, params).fetchall()
    return [RetrievedChunk(*row) for row in rows]


def _keyword_search(
    vector: str, tsquery: str, user_id: int, document_ids: list[int] | None, limit: int
) -> list[RetrievedChunk]:
    """Chunks containing any of the query words, best full-text rank first."""
    doc_filter = "AND c.document_id = ANY(%(docs)s)" if document_ids is not None else ""
    sql = f"""
        WITH q AS (SELECT to_tsquery('simple', %(tsq)s) AS query)
        SELECT {_COLUMNS}
        FROM chunks c JOIN documents d ON d.id = c.document_id, q
        WHERE c.user_id = %(user)s AND c.tsv @@ q.query {doc_filter}
        ORDER BY ts_rank_cd(c.tsv, q.query) DESC, c.id
        LIMIT %(limit)s
    """
    params = {"vec": vector, "tsq": tsquery, "user": user_id, "docs": document_ids, "limit": limit}
    with connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [RetrievedChunk(*row) for row in rows]


async def rerank_chunks(
    question: str,
    candidates: list[RetrievedChunk],
    top_k: int,
    *,
    client: Any | None = None,
) -> list[RetrievedChunk]:
    """Re-order candidates with the Voyage cross-encoder and keep the best ``top_k``.

    Raises:
        RerankServiceError: On API failure, timeout or a missing API key.
    """
    if not candidates:
        return []
    if client is None:
        if not settings.voyage_api_key:
            raise RerankServiceError("VOYAGE_API_KEY is not set.")
        client = voyageai.AsyncClient(
            api_key=settings.voyage_api_key,
            max_retries=RERANK_MAX_RETRIES,
            timeout=RERANK_TIMEOUT_SECONDS,
        )
    try:
        response = await client.rerank(
            question,
            [chunk.rerank_text for chunk in candidates],
            model=settings.rerank_model,
            top_k=top_k,
        )
    except Exception as exc:  # network, auth, rate limit after retries, timeout
        raise RerankServiceError(f"The reranking service failed: {exc}") from exc
    return [
        replace(candidates[item.index], rerank_score=float(item.relevance_score))
        for item in response.results
    ]


def _is_relevant(chunks: list[RetrievedChunk], reranked: bool, config: RetrievalConfig) -> bool:
    """Decide whether the best chunk is good enough to try answering from."""
    if not chunks:
        return False
    if reranked:
        return max(c.rerank_score or 0.0 for c in chunks) >= config.min_rerank_score
    return max(c.similarity for c in chunks) >= config.min_similarity


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


async def retrieve(
    question: str,
    user_id: int,
    *,
    document_ids: list[int] | None = None,
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    """Find the chunks of a user's documents that best answer a question.

    Args:
        question: The user's question.
        user_id: Whose documents to search.
        document_ids: Restrict the search to these documents (``None`` = all).
        config: Retrieval settings; defaults come from ``app.config``.

    Raises:
        EmbeddingServiceError: The question could not be embedded.
        DatabaseUnavailableError: The database is unreachable.

    A reranker failure does not raise: retrieval falls back to the fused order
    and reports ``reranked=False``, so the service keeps working during an outage.
    """
    cfg = config or RetrievalConfig()
    timings: dict[str, float] = {}

    started = time.perf_counter()
    vector = to_vector_literal(await embed_query(question))
    timings["embed"] = _elapsed_ms(started)

    started = time.perf_counter()
    tsquery = keyword_query(question) if cfg.use_keyword else None
    searches = [asyncio.to_thread(_vector_search, vector, user_id, document_ids, cfg.vector_k)]
    if tsquery:
        searches.append(
            asyncio.to_thread(_keyword_search, vector, tsquery, user_id, document_ids, cfg.keyword_k)
        )
    results = await asyncio.gather(*searches)
    vector_hits = results[0]
    keyword_hits = results[1] if len(results) > 1 else []
    timings["search"] = _elapsed_ms(started)

    by_id = {chunk.chunk_id: chunk for chunk in vector_hits + keyword_hits}
    scores = rrf_fuse(
        [[c.chunk_id for c in vector_hits], [c.chunk_id for c in keyword_hits]], cfg.rrf_k
    )
    ordered = sorted(by_id.values(), key=lambda c: (-scores[c.chunk_id], c.chunk_id))
    candidates = [replace(c, fused_score=scores[c.chunk_id]) for c in ordered][: cfg.rerank_k]

    reranked = False
    chunks = candidates[: cfg.final_k]
    if cfg.use_reranker and candidates:
        started = time.perf_counter()
        try:
            chunks = await rerank_chunks(question, candidates, cfg.final_k)
            reranked = True
        except RerankServiceError as exc:
            logger.warning("Reranking failed, using the fused order instead: %s", exc)
        timings["rerank"] = _elapsed_ms(started)

    result = RetrievalResult(
        chunks=chunks,
        candidates_considered=len(candidates),
        reranked=reranked,
        relevant=_is_relevant(chunks, reranked, cfg),
        timings_ms=timings,
    )
    logger.info(
        "Retrieved %d chunks (candidates=%d reranked=%s relevant=%s) timings=%s",
        len(chunks),
        len(candidates),
        reranked,
        result.relevant,
        timings,
    )
    return result

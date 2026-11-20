"""Tests for retrieval: search, fusion, reranking, isolation and thresholds.

Database-backed tests need ``docker compose up -d`` and are skipped otherwise.
No API calls are made: embeddings and the reranker are replaced by fakes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document

from app import retrieval
from app.ingestion import pipeline, store
from app.retrieval import (
    RerankServiceError,
    RetrievalConfig,
    RetrievedChunk,
    keyword_query,
    retrieve,
    rerank_chunks,
    rrf_fuse,
)
from tests.conftest import topic_embedder, topic_query_embedder

PLAIN = RetrievalConfig(use_keyword=False, use_reranker=False)
HYBRID = RetrievalConfig(use_keyword=True, use_reranker=False)


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use topic-based fake embeddings for both ingestion and questions."""
    monkeypatch.setattr(pipeline, "embed_texts", topic_embedder)
    monkeypatch.setattr(retrieval, "embed_query", topic_query_embedder)


def _make_doc(tmp_path: Path, name: str, sections: dict[str, str]) -> Path:
    """Create a Word file with one heading + paragraph per section."""
    document = Document()
    document.add_heading(name, level=1)
    for heading, body in sections.items():
        document.add_heading(heading, level=2)
        document.add_paragraph(body)
    path = tmp_path / f"{name}.docx"
    document.save(path)
    return path


def _ingest(user_id: int, path: Path) -> int:
    return asyncio.run(pipeline.ingest_file(path, path.name, user_id)).id


@pytest.fixture()
def handbook(db_user: int, tmp_path: Path) -> tuple[int, int]:
    """A stored document with refund, travel and security sections; returns (user, doc id)."""
    path = _make_doc(
        tmp_path,
        "handbook",
        {
            "Refunds": "Refund requests are processed within fourteen days of the claim. " * 12,
            "Travel": "Travel expenses need manager approval before any booking is made. " * 12
            + "Reference code ZX-4417 applies to hotel bookings.",
            "Security": "Security badges must be worn visibly inside every building. " * 12,
        },
    )
    return db_user, _ingest(db_user, path)


# ---------- pure functions ----------


def test_keyword_query_drops_stopwords_and_punctuation() -> None:
    assert keyword_query("What is the refund period?!") == "refund | period"


def test_keyword_query_cannot_inject_search_syntax() -> None:
    query = keyword_query("x'; DROP TABLE chunks; -- & ! | ( ) :*")
    assert query is not None
    assert all(ch.isalnum() or ch in " |_" for ch in query)


def test_keyword_query_is_none_when_nothing_useful_remains() -> None:
    assert keyword_query("What is it?") is None


def test_keyword_query_keeps_non_english_words_and_removes_duplicates() -> None:
    assert keyword_query("ریفنڈ ریفنڈ پالیسی") == "ریفنڈ | پالیسی"


def test_rrf_puts_chunks_found_by_both_searches_first() -> None:
    scores = rrf_fuse([[1, 2, 3], [3, 4]], k=60)
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked[0] == 3 and set(ranked) == {1, 2, 3, 4}


def _candidate(chunk_id: int, text: str = "text") -> RetrievedChunk:
    return RetrievedChunk(chunk_id, 1, "f.docx", "H", None, None, chunk_id, text, 0.5)


class FakeReranker:
    """Mimics ``voyageai.AsyncClient.rerank``: longer documents score higher."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.top_k_seen: int | None = None

    async def rerank(self, query, documents, model, top_k):  # noqa: ANN001
        self.top_k_seen = top_k
        if self.fail:
            raise TimeoutError("simulated timeout")
        order = sorted(range(len(documents)), key=lambda i: -len(documents[i]))[:top_k]
        return SimpleNamespace(
            results=[SimpleNamespace(index=i, relevance_score=1 - 0.1 * n) for n, i in enumerate(order)]
        )


def test_rerank_chunks_orders_by_the_reranker_and_keeps_top_k() -> None:
    candidates = [_candidate(1, "short"), _candidate(2, "a much longer passage here"), _candidate(3, "medium text")]
    client = FakeReranker()
    result = asyncio.run(rerank_chunks("q", candidates, 2, client=client))
    assert [c.chunk_id for c in result] == [2, 3]
    assert result[0].rerank_score == 1.0 and client.top_k_seen == 2


def test_rerank_failure_raises_rerank_service_error() -> None:
    with pytest.raises(RerankServiceError):
        asyncio.run(rerank_chunks("q", [_candidate(1)], 1, client=FakeReranker(fail=True)))


# ---------- database-backed ----------


def test_vector_search_ranks_the_matching_topic_first(handbook: tuple[int, int]) -> None:
    user_id, _ = handbook
    result = asyncio.run(retrieve("How long do refund requests take?", user_id, config=PLAIN))
    assert result.chunks[0].heading.endswith("Refunds")
    assert result.chunks[0].similarity > 0.95
    assert not result.reranked and result.relevant


def test_every_chunk_carries_a_similarity_score(handbook: tuple[int, int]) -> None:
    user_id, _ = handbook
    result = asyncio.run(retrieve("travel approval", user_id, config=HYBRID))
    assert result.chunks and all(isinstance(c.similarity, float) for c in result.chunks)
    assert len(result.chunks) <= RetrievalConfig().final_k


def test_keyword_search_finds_an_exact_code_that_embeddings_cannot(handbook: tuple[int, int]) -> None:
    user_id, _ = handbook
    result = asyncio.run(retrieve("What is code ZX-4417?", user_id, config=HYBRID))
    assert "ZX-4417" in result.chunks[0].text


def test_reranker_reorders_and_adds_scores(
    handbook: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id, _ = handbook
    reranker = FakeReranker()

    async def rerank_with_fake(question, candidates, top_k, **_):  # noqa: ANN001
        return await rerank_chunks(question, candidates, top_k, client=reranker)

    monkeypatch.setattr(retrieval, "rerank_chunks", rerank_with_fake)
    result = asyncio.run(retrieve("refund", user_id, config=RetrievalConfig(use_reranker=True)))
    assert result.reranked and "rerank" in result.timings_ms
    assert all(c.rerank_score is not None for c in result.chunks)
    scores = [c.rerank_score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_reranker_outage_falls_back_to_the_fused_order(
    handbook: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id, _ = handbook

    async def broken(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RerankServiceError("simulated outage")

    monkeypatch.setattr(retrieval, "rerank_chunks", broken)
    result = asyncio.run(retrieve("refund", user_id, config=RetrievalConfig(use_reranker=True)))
    assert not result.reranked and result.chunks


def test_unrelated_question_is_flagged_as_not_answerable(handbook: tuple[int, int]) -> None:
    user_id, _ = handbook
    result = asyncio.run(retrieve("Explain quantum chromodynamics", user_id, config=PLAIN))
    assert result.chunks and not result.relevant  # best similarity is only about 0.1


def test_other_users_documents_are_never_returned(
    handbook: tuple[int, int], user_factory: Callable[[], int], tmp_path: Path
) -> None:
    owner, _ = handbook
    outsider = user_factory()
    _ingest(outsider, _make_doc(tmp_path, "secret", {"Refunds": "Secret refund formula alpha. " * 20}))
    mine = asyncio.run(retrieve("refund", owner, config=HYBRID))
    theirs = asyncio.run(retrieve("refund", outsider, config=HYBRID))
    assert {c.filename for c in mine.chunks} == {"handbook.docx"}
    assert {c.filename for c in theirs.chunks} == {"secret.docx"}


def test_search_can_be_limited_to_chosen_documents(
    handbook: tuple[int, int], tmp_path: Path
) -> None:
    user_id, handbook_id = handbook
    other_id = _ingest(user_id, _make_doc(tmp_path, "other", {"Refunds": "Other refund rules apply. " * 20}))
    only_other = asyncio.run(retrieve("refund", user_id, document_ids=[other_id], config=HYBRID))
    assert {c.document_id for c in only_other.chunks} == {other_id}
    nothing = asyncio.run(retrieve("refund", user_id, document_ids=[], config=HYBRID))
    assert nothing.chunks == []


def test_deleted_documents_are_never_retrieved(handbook: tuple[int, int]) -> None:
    user_id, document_id = handbook
    assert asyncio.run(retrieve("refund", user_id, config=HYBRID)).chunks
    store.delete_document(user_id, document_id)
    result = asyncio.run(retrieve("refund ZX-4417", user_id, config=HYBRID))
    assert result.chunks == [] and not result.relevant


def test_asking_before_uploading_anything_is_handled(db_user: int) -> None:
    result = asyncio.run(retrieve("What is the refund policy?", db_user))
    assert result.chunks == [] and not result.relevant and not result.reranked


def test_timings_are_reported_for_each_stage(handbook: tuple[int, int]) -> None:
    user_id, _ = handbook
    result = asyncio.run(retrieve("refund", user_id, config=HYBRID))
    assert {"embed", "search"} <= set(result.timings_ms)

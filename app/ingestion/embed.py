"""Embedding client (Voyage AI).

Texts are sent in batches that run concurrently, with a timeout and automatic
retries on every request. Documents and queries use different ``input_type``
values, because the model embeds them slightly differently to improve retrieval.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import voyageai

from app.config import settings
from app.ingestion.errors import EmbeddingServiceError

logger = logging.getLogger(__name__)

BATCH_SIZE = 64  # chunks per request: well under the API's input limits
MAX_CONCURRENT_BATCHES = 4  # parallel requests, to stay clear of rate limits
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3  # the client retries rate-limit and server errors with backoff


def _make_client() -> voyageai.AsyncClient:
    """Create the async Voyage client from the configured API key."""
    if not settings.voyage_api_key:
        raise EmbeddingServiceError("VOYAGE_API_KEY is not set.")
    return voyageai.AsyncClient(
        api_key=settings.voyage_api_key,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


async def embed_texts(
    texts: list[str],
    input_type: str = "document",
    *,
    client: Any | None = None,
) -> list[list[float]]:
    """Embed texts, returning one vector per text in the same order.

    Args:
        texts: Strings to embed.
        input_type: ``"document"`` for stored chunks, ``"query"`` for questions.
        client: Optional client (used by tests); defaults to a real Voyage client.

    Raises:
        EmbeddingServiceError: On API failure, timeout or unexpected output.
    """
    if not texts:
        return []
    api = client or _make_client()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

    async def embed_batch(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            try:
                result = await api.embed(
                    batch,
                    model=settings.embedding_model,
                    input_type=input_type,
                    output_dimension=settings.embedding_dimensions,
                )
            except Exception as exc:  # network, auth, rate limit after retries, timeout
                raise EmbeddingServiceError(f"The embedding service failed: {exc}") from exc
            return result.embeddings

    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    # gather() returns results in the order the batches were given.
    batch_results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
    vectors = [vector for batch in batch_results for vector in batch]

    if len(vectors) != len(texts):
        raise EmbeddingServiceError("The embedding service returned the wrong number of vectors.")
    if any(len(vector) != settings.embedding_dimensions for vector in vectors):
        raise EmbeddingServiceError("The embedding service returned vectors of the wrong size.")
    logger.info("Embedded %d texts in %d batches", len(texts), len(batches))
    return vectors


async def embed_query(query: str, *, client: Any | None = None) -> list[float]:
    """Embed one user question."""
    (vector,) = await embed_texts([query], "query", client=client)
    return vector

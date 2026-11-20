"""Central configuration for the RAG service.

Every setting and constant lives here. Secrets are read from environment
variables only (a local ``.env`` file is loaded in development) and are never
hard-coded or logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    """Return an environment variable, treating empty strings as unset."""
    value = os.getenv(name)
    return value if value else default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings, evaluated once at start-up."""

    # --- Secrets and connections (environment only, never hard-coded) ---
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    # Only needed when the Anthropic key is not scoped to a single workspace.
    anthropic_workspace_id: str | None = field(
        default_factory=lambda: _env("ANTHROPIC_WORKSPACE_ID")
    )
    voyage_api_key: str | None = field(default_factory=lambda: _env("VOYAGE_API_KEY"))
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
        or ""
    )

    # --- Models ---
    embedding_model: str = "voyage-4"
    embedding_dimensions: int = 1024
    llm_model: str = "claude-sonnet-5-5"
    fast_llm_model: str = "claude-haiku-5-5"

    # --- Upload limits ---
    max_upload_mb: int = 25
    max_pages: int = 300
    allowed_extensions: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"

    # --- Chunking (starting values; the evaluation decides the final ones) ---
    chunk_target_tokens: int = 500  # aim for chunks of about this size
    chunk_max_tokens: int = 800  # hard ceiling for any single chunk
    chunk_overlap_tokens: int = 60  # text repeated when a section is cut mid-way
    chunk_min_tokens: int = 120  # smaller sections are merged with their neighbour

    # --- Retrieval (starting values; the evaluation decides the final ones) ---
    rerank_model: str = "rerank-3"  # Voyage reranker; "rerank-2.5" is the alternative
    vector_candidates: int = 30  # chunks taken from vector search
    keyword_candidates: int = 30  # chunks taken from keyword search
    rerank_candidates: int = 40  # merged candidates sent to the reranker
    final_top_k: int = 6  # chunks finally given to the LLM
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    min_similarity: float = 0.20  # without a reranker: below this = "not in the documents"
    min_rerank_score: float = 0.10  # with a reranker: below this = "not in the documents"


settings = Settings()

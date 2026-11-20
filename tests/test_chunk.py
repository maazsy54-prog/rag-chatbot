"""Unit tests for the chunker (no network or API calls needed)."""

from __future__ import annotations

from app.ingestion.chunk import Chunk, chunk_pages, count_tokens
from app.ingestion.parse import ParsedPage


def _sentences(prefix: str, count: int) -> str:
    """Build a long paragraph of distinct sentences."""
    return " ".join(f"{prefix} sentence number {i} explains one separate idea clearly." for i in range(count))


def test_chunks_never_exceed_the_maximum() -> None:
    pages = [ParsedPage(1, f"# Report\n\n{_sentences('Alpha', 400)}")]
    chunks = chunk_pages(pages, target_tokens=200, max_tokens=320, overlap_tokens=20)
    assert len(chunks) > 3
    # Overlap can add a few tokens on top of the ceiling, so allow a small margin.
    assert all(chunk.token_count <= 320 + 40 for chunk in chunks)


def test_heading_path_and_pages_are_kept() -> None:
    pages = [
        ParsedPage(1, f"# Contract\n\n## Payment\n\n{_sentences('Pay', 30)}"),
        ParsedPage(2, f"## Termination\n\n{_sentences('End', 30)}"),
    ]
    chunks = chunk_pages(pages, target_tokens=300, max_tokens=500, min_tokens=50)
    by_heading = {chunk.heading for chunk in chunks}
    assert "Contract > Payment" in by_heading
    assert "Contract > Termination" in by_heading
    termination = [c for c in chunks if c.heading == "Contract > Termination"]
    assert all(c.page_start == 2 and c.page_end == 2 for c in termination)


def test_sections_are_not_mixed_when_large_enough() -> None:
    pages = [ParsedPage(1, f"# A\n\n{_sentences('First', 40)}\n\n# B\n\n{_sentences('Second', 40)}")]
    chunks = chunk_pages(pages, target_tokens=500, max_tokens=800, min_tokens=50)
    for chunk in chunks:
        assert not ("First" in chunk.text and "Second" in chunk.text)


def test_tiny_sections_are_merged_with_their_title() -> None:
    pages = [ParsedPage(1, "# A\n\nShort note one here.\n\n# B\n\nShort note two here.")]
    chunks = chunk_pages(pages, target_tokens=500, max_tokens=800, min_tokens=120)
    assert len(chunks) == 1
    assert "B" in chunks[0].text and "Short note two here." in chunks[0].text


def test_small_table_stays_in_one_chunk() -> None:
    table = "| Item | Price |\n| --- | --- |\n| Tea | 3 |\n| Coffee | 4 |"
    chunks = chunk_pages([ParsedPage(1, f"# Menu\n\n{table}")])
    assert len(chunks) == 1
    assert table in chunks[0].text


def test_large_table_is_split_with_header_repeated() -> None:
    rows = "\n".join(f"| Item {i} | {i * 7} |" for i in range(300))
    table = f"| Item | Price |\n| --- | --- |\n{rows}"
    chunks = chunk_pages([ParsedPage(1, f"# Menu\n\n{table}")], target_tokens=200, max_tokens=300)
    assert len(chunks) > 1
    assert all(chunk.text.startswith("| Item | Price |") for chunk in chunks)


def test_overlap_repeats_the_end_of_the_previous_chunk() -> None:
    pages = [ParsedPage(1, f"# Long\n\n{_sentences('Beta', 300)}")]
    chunks = chunk_pages(pages, target_tokens=200, max_tokens=320, overlap_tokens=30)
    first_tail = " ".join(chunks[0].text.split()[-5:])
    assert first_tail in chunks[1].text


def test_no_overlap_across_sections() -> None:
    pages = [ParsedPage(1, f"# A\n\n{_sentences('First', 40)}\n\n# B\n\n{_sentences('Second', 40)}")]
    chunks = chunk_pages(pages, target_tokens=300, max_tokens=500, min_tokens=50)
    second_section = [c for c in chunks if c.heading == "B"]
    assert second_section and "First" not in second_section[0].text


def test_chunking_is_deterministic_and_indexed() -> None:
    pages = [ParsedPage(1, f"# Doc\n\n{_sentences('Gamma', 200)}")]
    first = chunk_pages(pages, target_tokens=250, max_tokens=400)
    second = chunk_pages(pages, target_tokens=250, max_tokens=400)
    assert first == second
    assert [c.index for c in first] == list(range(len(first)))


def test_embed_text_prefixes_the_heading() -> None:
    chunk = Chunk(0, "Payment is due in 30 days.", "Contract > Payment", 3, 3, count_tokens("x"))
    assert chunk.embed_text.startswith("Contract > Payment")
    assert chunk.embed_text.endswith("Payment is due in 30 days.")


def test_empty_input_gives_no_chunks() -> None:
    assert chunk_pages([ParsedPage(1, "")]) == []


def test_text_without_spaces_is_still_split() -> None:
    pages = [ParsedPage(None, "x" * 20000)]
    chunks = chunk_pages(pages, target_tokens=200, max_tokens=300)
    assert len(chunks) > 1

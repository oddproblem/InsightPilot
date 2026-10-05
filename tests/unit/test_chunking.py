"""Unit tests for document chunking."""

from app.rag.chunking import chunk_text


def test_chunk_text_empty() -> None:
    assert chunk_text("", source="empty.txt") == []
    assert chunk_text("   ", source="spaces.txt") == []


def test_chunk_text_paragraphs() -> None:
    text = "Paragraph 1: Revenue was $10M.\n\nParagraph 2: Net income was $2M."
    chunks = chunk_text(text, source="report.pdf", chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 2
    assert all(c.source == "report.pdf" for c in chunks)
    assert "Revenue" in chunks[0].content

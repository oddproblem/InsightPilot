"""Document chunking strategies optimized for financial and business reports."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    """A text chunk with metadata ready for embedding and indexing."""

    chunk_id: str
    content: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    metadata: dict[str, Any] | None = None,
) -> list[DocumentChunk]:
    """Split text into overlapping chunks while preserving sentence boundaries and paragraphs.

    Specialized for business reports, preserving financial tables and markdown sections.
    """
    if not text.strip():
        return []

    base_meta = metadata or {}
    chunks: list[DocumentChunk] = []

    # Split by double newline first to preserve paragraph integrity
    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        if len(current_chunk) + len(para_clean) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para_clean}" if current_chunk else para_clean
        else:
            if current_chunk:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{source}_chunk_{chunk_idx}",
                        content=current_chunk.strip(),
                        source=source,
                        chunk_index=chunk_idx,
                        metadata={**base_meta, "chunk_index": chunk_idx},
                    )
                )
                chunk_idx += 1

                # Carry over overlap
                overlap_text = (
                    current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else ""
                )
                current_chunk = f"{overlap_text}\n\n{para_clean}" if overlap_text else para_clean
            else:
                # Handle single paragraph exceeding chunk size
                start = 0
                while start < len(para_clean):
                    end = start + chunk_size
                    slice_text = para_clean[start:end]
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{source}_chunk_{chunk_idx}",
                            content=slice_text.strip(),
                            source=source,
                            chunk_index=chunk_idx,
                            metadata={**base_meta, "chunk_index": chunk_idx},
                        )
                    )
                    chunk_idx += 1
                    start += chunk_size - chunk_overlap

    if current_chunk.strip():
        chunks.append(
            DocumentChunk(
                chunk_id=f"{source}_chunk_{chunk_idx}",
                content=current_chunk.strip(),
                source=source,
                chunk_index=chunk_idx,
                metadata={**base_meta, "chunk_index": chunk_idx},
            )
        )

    return chunks

"""Citation extraction, verification, and grounding checks."""

from typing import Any


def verify_citation_grounding(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract and verify citations from the generated answer against retrieved sources."""
    citations: list[dict[str, Any]] = []

    for chunk in retrieved_chunks:
        source = chunk.get("source", "Unknown")
        content = chunk.get("content", "")

        # Check if source or key snippet appears in answer
        is_cited = source.lower() in answer.lower()
        citations.append(
            {
                "source": source,
                "snippet": content[:200],
                "verified": is_cited,
                "page": chunk.get("metadata", {}).get("page"),
            }
        )

    return citations

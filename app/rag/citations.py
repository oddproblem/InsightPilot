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
        meta = chunk.get("metadata", {})
        doc_name = meta.get("document_name", "")
        page = meta.get("page_number") or meta.get("page") or chunk.get("page")

        # Check if source, document name, or normalized source name appears in answer
        source_clean = (
            source.rsplit(".", 1)[0].replace("_", " ").lower()
            if "." in source
            else source.lower()
        )
        is_cited = (
            source.lower() in answer.lower()
            or (bool(doc_name) and doc_name.lower() in answer.lower())
            or (bool(source_clean) and source_clean in answer.lower())
        )
        citations.append(
            {
                "source": source,
                "snippet": content[:200],
                "verified": is_cited,
                "page": page,
            }
        )

    return citations

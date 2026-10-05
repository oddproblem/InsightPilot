"""Generation quality and groundedness evaluation."""

from typing import Any


def evaluate_faithfulness(answer: str, context_chunks: list[dict[str, Any]]) -> float:
    """Assess whether key numbers and statements in the answer exist in the context."""
    if not answer.strip():
        return 0.0

    if not context_chunks:
        # If no context provided, faithfulness cannot be proven from context
        return 0.0

    combined_context = " ".join(c.get("content", "") for c in context_chunks).lower()

    # Simple lexical containment heuristic for initial evaluation harness
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words:
        return 1.0

    matches = sum(1 for w in words if w in combined_context)
    return round(matches / len(words), 4)

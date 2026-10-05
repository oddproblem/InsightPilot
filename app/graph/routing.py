"""Query analysis and routing logic for InsightPilot agent workflow."""

import re
from typing import Literal

RouteType = Literal["document_search", "web_search", "calculator", "direct"]

# Pattern heuristics for fast pre-routing / fallback routing
CALCULATOR_PATTERNS = [
    r"\bcagr\b",
    r"\bgrowth\s*rate\b",
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bpercentage\s*change\b",
    r"\bformula\b",
    r"\bmultiply\b",
    r"\bdivide\b",
    r"\bvariance\s+between\b",
]

WEB_PATTERNS = [
    r"\btoday\b",
    r"\blatest\s*news\b",
    r"\bcurrent\s*stock\s*price\b",
    r"\bmarket\s*sentiment\b",
    r"\bcompetitor\s*news\b",
    r"\brecently\b",
]

DOC_PATTERNS = [
    r"\brevenue\b",
    r"\bfy20\d\d\b",
    r"\bquarter\b",
    r"\b10-k\b",
    r"\b10-q\b",
    r"\boperating\s*margin\b",
    r"\bebitda\b",
    r"\buploaded\b",
    r"\bdocument\b",
    r"\bevidence\b",
    r"\breport\b",
]


def classify_query_route(query: str) -> RouteType:
    """Classify user query into an appropriate route using pattern heuristics.

    Used directly or as deterministic fallback for LLM query classification.
    """
    cleaned = query.strip().lower()

    if not cleaned:
        return "direct"

    # Check for calculation requests
    for pattern in CALCULATOR_PATTERNS:
        if re.search(pattern, cleaned):
            return "calculator"

    # Check for live web search queries
    for pattern in WEB_PATTERNS:
        if re.search(pattern, cleaned):
            return "web_search"

    # Check for document retrieval queries
    for pattern in DOC_PATTERNS:
        if re.search(pattern, cleaned):
            return "document_search"

    # Default to direct response for generic conversational messages
    return "direct"

"""Output security guard: PII detection, secret sanitization, and leak defense."""

import re

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    (r"\b4[0-9]{12}(?:[0-9]{3})?\b", "[REDACTED_CC]"),
    (r"sk-[a-zA-Z0-9]{20,}", "[REDACTED_API_KEY]"),
]


def sanitize_output(output_text: str) -> tuple[str, list[str]]:
    """Sanitize model output to remove PII, leaked credentials, or sensitive tokens."""
    sanitized = output_text
    flags: list[str] = []

    for pattern, replacement in PII_PATTERNS:
        if re.search(pattern, sanitized):
            flags.append("PII_OR_SECRET_REDACTED")
            sanitized = re.sub(pattern, replacement, sanitized)

    return sanitized, flags

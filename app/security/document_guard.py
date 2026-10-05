"""Document security guard: indirect injection and embedded malicious payload detection."""

import re

INDIRECT_INJECTION_PATTERNS = [
    r"\[system\s*override\]",
    r"<!--\s*ai\s*instruction:",
    r"disregard\s+document\s+contents\s+and\s+say",
    r"secret_key\s*=",
    r"aws_access_key_id",
]


def check_document_safety(content: str) -> list[str]:
    """Scan uploaded document text for indirect prompt injection or embedded secrets."""
    flags: list[str] = []
    lowered = content.lower()

    for pattern in INDIRECT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            flags.append(f"INDIRECT_INJECTION_FLAG: {pattern}")

    return flags

"""Input security guard: prompt injection and adversarial prompt detection."""

import re

# High-risk adversarial phrases and prompt-injection patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt\s+override",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"bypass\s+all\s+safety\s+filters",
    r"dan\s+mode",
    r"output\s+all\s+system\s+prompts",
    r"reveal\s+your\s+instructions",
]


def check_input_safety(user_input: str) -> list[str]:
    """Inspect user input query for prompt injection and adversarial attacks."""
    flags: list[str] = []
    lowered = user_input.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            flags.append(f"PROMPT_INJECTION_DETECTED: {pattern}")

    return flags

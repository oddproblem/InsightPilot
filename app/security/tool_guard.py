"""Tool execution security guard: validates tool inputs and parameters."""

from typing import Any


def validate_tool_call(tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
    """Validate tool input arguments before invoking tool execution."""
    if tool_name == "calculator":
        expression = str(args.get("expression", ""))
        # Only allow safe mathematical characters
        allowed_chars = set("0123456789+-*/().,% eE")
        if not all(c in allowed_chars for c in expression):
            return False, "Unsafe character in calculator expression."
        if len(expression) > 200:
            return False, "Expression length exceeds safety limit."

    elif tool_name == "web_search":
        query = str(args.get("query", ""))
        if len(query) > 500:
            return False, "Search query exceeds length limit."

    return True, "Valid"

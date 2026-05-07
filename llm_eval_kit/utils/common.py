"""
Common utility functions for LLM evaluation.
"""

import json
from typing import Any, Dict


def truncate_text(text: str, max_len: int = 100) -> str:
    """Truncate text for display."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def safe_json_loads(text: str) -> Dict[str, Any]:
    """Safely parse JSON, returning empty dict on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}

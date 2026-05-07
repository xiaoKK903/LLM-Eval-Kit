"""
Unified cost calculation for LLM API calls.

Supports both CNY (Chinese Yuan) and USD pricing.
CNY pricing: per million tokens (官方定价)
USD pricing: per 1K tokens (OpenAI-compatible convention)
"""

from typing import Dict, Optional

CNY_PRICING: Dict[str, Dict[str, float]] = {
    "qwen-turbo": {"input": 0.3, "output": 0.6},
    "qwen-plus": {"input": 0.8, "output": 2.0},
    "qwen-max": {"input": 2.0, "output": 4.0},
    "deepseek-chat": {"input": 1.0, "output": 2.0},
    "deepseek-coder": {"input": 1.0, "output": 2.0},
    "gpt-3.5-turbo": {"input": 1.5, "output": 2.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 15.0, "output": 30.0},
}

USD_PRICING: Dict[str, Dict[str, float]] = {
    "qwen-turbo": {"input": 0.002, "output": 0.008},
    "qwen-plus": {"input": 0.01, "output": 0.02},
    "qwen-max": {"input": 0.04, "output": 0.08},
    "deepseek-chat": {"input": 0.0007, "output": 0.0028},
    "deepseek-coder": {"input": 0.0007, "output": 0.0028},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}

DEFAULT_CNY = {"input": 1.0, "output": 2.0}
DEFAULT_USD = {"input": 0.0015, "output": 0.002}


def _find_pricing(model_name: str, pricing_dict: Dict[str, Dict[str, float]],
                  default: Dict[str, float]) -> Dict[str, float]:
    for key in pricing_dict:
        if key in model_name.lower():
            return pricing_dict[key]
    return default


def calculate_cost_cny(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in CNY (¥) based on per-million-token pricing."""
    p = _find_pricing(model_name, CNY_PRICING, DEFAULT_CNY)
    input_cost = (prompt_tokens / 1_000_000) * p["input"]
    output_cost = (completion_tokens / 1_000_000) * p["output"]
    return input_cost + output_cost


def calculate_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD ($) based on per-1K-token pricing."""
    p = _find_pricing(model_name, USD_PRICING, DEFAULT_USD)
    input_cost = (prompt_tokens / 1000) * p["input"]
    output_cost = (completion_tokens / 1000) * p["output"]
    return input_cost + output_cost


def format_cost(cost: float, currency: str = "CNY") -> str:
    """Format cost for display. Auto-converts tiny values."""
    if cost < 0.01:
        fen = cost * 100
        return f"{fen:.2f}分"
    if currency == "CNY":
        return f"¥{cost:.4f}"
    return f"${cost:.4f}"

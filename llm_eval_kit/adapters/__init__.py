"""
Model adapters for different LLM providers.

This module provides adapters for various LLM APIs, with a focus on
OpenAI-compatible interfaces that cover most modern LLM providers.
"""

from .openai_compatible import OpenAICompatibleHttpxAdapter
from .multi_provider_adapter import MultiProviderAdapter

__all__ = ["OpenAICompatibleHttpxAdapter", "MultiProviderAdapter"]
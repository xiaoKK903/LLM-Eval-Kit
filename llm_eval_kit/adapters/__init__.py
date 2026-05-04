"""
Model adapters for different LLM providers.

This module provides adapters for various LLM APIs, with a focus on
OpenAI-compatible interfaces that cover most modern LLM providers.
"""

from .openai_compatible import OpenAICompatibleAdapter

__all__ = ["OpenAICompatibleAdapter"]
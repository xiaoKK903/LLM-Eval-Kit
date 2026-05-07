"""
LLM-Eval-Kit: A lightweight LLM evaluation toolkit for business scenarios.

This package provides tools for evaluating LLM performance in business contexts,
including prompt changes, model comparisons, and RAG system effectiveness.
"""

from .client import EvalClient, run_evaluation
from .dataset.loader import DatasetLoader
from .adapters.openai_compatible import OpenAICompatibleAdapter
from .reporter.console import ConsoleReporter

__version__ = "0.1.0"
__all__ = ["EvalClient", "run_evaluation", "DatasetLoader", "OpenAICompatibleAdapter", "ConsoleReporter"]
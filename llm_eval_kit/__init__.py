"""
LLM-Eval-Kit: 中文轻量业务 LLM 评测工具箱

Usage:
    from llm_eval_kit import Evaluator

    evaluator = Evaluator()
    result = await evaluator.run(
        models=[{...}],
        data_path="data.jsonl"
    )
"""

from .core.evaluator import Evaluator, EvalConfig
from .adapters import OpenAICompatibleAdapter
from .scorers import RuleScorer, QualityScorer, LLMJudgeScorer
from .dataset import DatasetLoader
from .reporter import ConsoleReporter

__version__ = "0.1.0"

__all__ = [
    "Evaluator", "EvalConfig",
    "OpenAICompatibleAdapter",
    "RuleScorer", "QualityScorer", "LLMJudgeScorer",
    "DatasetLoader",
    "ConsoleReporter",
]

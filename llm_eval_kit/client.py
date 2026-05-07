"""
Legacy module — re-exports Evaluator from new core location.

Kept for backward compatibility with existing examples.
"""

from .core.evaluator import Evaluator, EvalConfig

EvaluationConfig = EvalConfig

__all__ = ["Evaluator", "EvaluationConfig"]


async def run_evaluation(data_path: str, model_config: dict, verbose: bool = False,
                         max_samples: int = None, **kwargs):
    """Legacy convenience wrapper — runs a single-model evaluation."""
    evaluator = Evaluator()
    result = await evaluator.run(
        models=[model_config],
        data_path=data_path,
        max_samples=max_samples,
        concurrency=1,
    )
    # Return flat list of EvaluationResult for backward compat
    return result.model_comparisons[0] if result.model_comparisons else []

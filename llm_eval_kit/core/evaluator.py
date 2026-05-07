"""
Central evaluator orchestrating model calls, scoring, and reporting.

Usage:
    evaluator = Evaluator()
    result = await evaluator.run(
        models=[model_config1, model_config2],
        data_path="data.jsonl"
    )
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List
from asyncio import Semaphore
from dataclasses import dataclass

from ..adapters.openai_compat import OpenAICompatibleAdapter
from ..adapters.base import ModelResponse
from ..dataset.loader import DatasetLoader, EvaluationSample
from ..scorers.base import BaseScorer, ScoreResult
from ..scorers.rule_scorer import RuleScorer
from ..scorers.llm_judge import LLMJudgeScorer
from ..reporter.models import EvaluationResult
from ..reporter.comparator import ModelComparator, ComparisonResult
from ..reporter.console_reporter import ConsoleReporter
from ..utils.cost_calc import calculate_cost_cny


@dataclass
class EvalConfig:
    data_path: str
    max_samples: Optional[int] = None
    concurrency: int = 5
    output_path: Optional[str] = None


class Evaluator:
    """Central orchestrator for LLM evaluation."""

    def __init__(self):
        self.comparator = ModelComparator()
        self.reporter = ConsoleReporter()

    async def run(
        self,
        models: List[Dict[str, Any]],
        data_path: str,
        scorer: Optional[BaseScorer] = None,
        max_samples: Optional[int] = None,
        concurrency: int = 5,
        output_path: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Run full evaluation pipeline.

        Args:
            models: List of model configs, each with base_url, api_key, model
            data_path: Path to JSONL data file
            scorer: Scorer to use (default: RuleScorer)
            max_samples: Limit number of samples
            concurrency: Max concurrent requests per model
            output_path: Optional path to save JSON report

        Returns:
            ComparisonResult with all model comparisons
        """
        scorer = scorer or RuleScorer()

        loader = DatasetLoader(data_path)
        samples = loader.load()
        if max_samples:
            samples = samples[:max_samples]

        print(f"\n{'='*60}")
        print(f"  LLM-Eval-Kit: {len(samples)} samples × {len(models)} models")
        print(f"{'='*60}\n")

        semaphore = Semaphore(concurrency)
        all_results: List[EvaluationResult] = []

        for model_cfg in models:
            model_name = model_cfg.get("model", "unknown")
            print(f"  Evaluating {model_name}...")

            adapter = OpenAICompatibleAdapter(model_cfg)
            tasks = [
                self._evaluate_one(adapter, sample, scorer, semaphore)
                for sample in samples
            ]
            model_results = await asyncio.gather(*tasks, return_exceptions=True)

            for r in model_results:
                if isinstance(r, Exception):
                    print(f"    Error: {r}")
                else:
                    all_results.append(r)

            await adapter.close()

        comparison = self.comparator.compare_models(all_results)

        self.reporter.print_comparison(comparison)

        if output_path:
            self._save_json(comparison, output_path)

        return comparison

    async def _evaluate_one(
        self,
        adapter: OpenAICompatibleAdapter,
        sample: EvaluationSample,
        scorer: BaseScorer,
        semaphore: Semaphore,
    ) -> EvaluationResult:
        async with semaphore:
            resp = await adapter.generate(sample.question)

            scoring = await scorer.score(sample.question, resp.text, sample.reference or "")

            total_tokens = resp.token_usage.get("total_tokens", 0)
            prompt_tokens = resp.token_usage.get("prompt_tokens", 0)
            completion_tokens = resp.token_usage.get("completion_tokens", 0)

            cost = calculate_cost_cny(resp.model, prompt_tokens, completion_tokens)

            return EvaluationResult(
                sample_id=sample.id,
                question=sample.question,
                response=resp.text,
                latency=resp.latency,
                token_usage=resp.token_usage,
                model=resp.model,
                cost=cost,
                scoring_result=scoring.to_dict() if isinstance(scoring, ScoreResult) else scoring,
            )

    @staticmethod
    def _save_json(comparison: ComparisonResult, path: str):
        data = {
            "best_overall": {
                "model": comparison.best_overall_model,
                "score": comparison.best_overall_score,
            },
            "fastest": {
                "model": comparison.fastest_model,
                "latency": comparison.fastest_latency,
            },
            "cheapest": {
                "model": comparison.cheapest_model,
                "cost": comparison.cheapest_cost,
            },
            "best_value": {
                "model": comparison.best_value_model,
                "ratio": comparison.best_value_ratio,
            },
            "models": [
                {
                    "name": m.model_name,
                    "avg_score": m.avg_score,
                    "avg_latency": m.avg_latency,
                    "total_tokens": m.total_tokens,
                    "total_cost": m.total_cost,
                    "cost_per_sample": m.cost_per_sample,
                    "score_per_cost": m.score_per_cost,
                    "success_rate": m.success_rate,
                }
                for m in comparison.model_comparisons
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n  JSON report saved: {path}")

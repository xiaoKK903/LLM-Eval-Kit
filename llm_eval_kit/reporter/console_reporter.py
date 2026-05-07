"""
Console reporter for evaluation results.

Displays comparison tables and individual scores in the terminal.
"""

from typing import List, Dict, Any, Optional

from .models import EvaluationResult
from .comparator import ComparisonResult, ModelComparison


class ConsoleReporter:
    """Report evaluation results to the terminal."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def print_comparison(self, result: ComparisonResult):
        """Print model comparison table."""
        if not result.model_comparisons:
            print("No data to report.")
            return

        print(f"\n{'='*70}")
        print(f"  评测结果对比")
        print(f"{'='*70}\n")

        header = f"{'模型':<20} {'综合分':<10} {'延迟(s)':<10} {'总Token':<10} {'总成本(¥)':<12} {'性价比':<10}"
        print(header)
        print("-" * 72)

        for m in result.model_comparisons:
            name = m.model_name[:18]
            score_str = f"{m.avg_score:.4f}"
            latency_str = f"{m.avg_latency:.2f}"
            token_str = f"{m.total_tokens}"
            cost_str = self._fmt_cost(m.total_cost)
            value_str = f"{m.score_per_cost:.2f}" if m.score_per_cost > 0 else "N/A"
            print(f"{name:<20} {score_str:<10} {latency_str:<10} {token_str:<10} {cost_str:<12} {value_str:<10}")

        print("-" * 72)

        icons = {"★": result.best_overall_model, "⚡": result.fastest_model,
                 "¥": result.cheapest_model, "◆": result.best_value_model}
        print(f"\n  最佳综合:  ★ {result.best_overall_model} ({result.best_overall_score:.4f})")
        print(f"  最快响应:  ⚡ {result.fastest_model} ({result.fastest_latency:.2f}s)")
        print(f"  最低成本:  ¥ {result.cheapest_model} ({self._fmt_cost(result.cheapest_cost)}/条)")
        print(f"  最高性价比: ◆ {result.best_value_model} ({result.best_value_ratio:.2f})")
        print()

    def print_detailed(self, results: List[EvaluationResult]):
        """Print per-sample details."""
        if not results:
            return

        print(f"\n{'='*70}")
        print(f"  逐条详情")
        print(f"{'='*70}\n")

        for r in results:
            score_str = ""
            if r.scoring_result:
                total = r.scoring_result.get("total_score", 0)
                score_str = f" | 评分: {total:.4f}"

            cost_str = f" | 成本: {self._fmt_cost(r.cost)}" if r.cost > 0 else ""

            print(f"  [{r.sample_id}] {r.model}")
            print(f"  耗时: {r.latency:.2f}s{score_str}{cost_str}")
            print(f"  Token: {r.token_usage}")
            if r.response:
                resp_preview = r.response[:120].replace("\n", " ")
                print(f"  回答: {resp_preview}...")
            print()

    def _fmt_cost(self, cost: float) -> str:
        if cost < 0.01:
            fen = cost * 100
            return f"{fen:.2f}分"
        return f"¥{cost:.4f}"

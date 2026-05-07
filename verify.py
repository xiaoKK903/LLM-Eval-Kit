"""Final integration verification"""
import asyncio
from llm_eval_kit import Evaluator, EvalConfig
from llm_eval_kit import OpenAICompatibleAdapter
from llm_eval_kit import RuleScorer, QualityScorer, LLMJudgeScorer
from llm_eval_kit import DatasetLoader
from llm_eval_kit import ConsoleReporter
from llm_eval_kit.adapters import BaseAdapter, ModelResponse
from llm_eval_kit.scorers import BaseScorer, ScoreResult
from llm_eval_kit.utils import calculate_cost_cny, format_cost
from llm_eval_kit.reporter import ModelComparator, ComparisonResult
from llm_eval_kit.reporter.models import EvaluationResult
from llm_eval_kit.reporter.html_reporter import HtmlReporter
from llm_eval_kit import __version__

print(f"[OK] __version__ = {__version__}")

# Instantiate all core classes
evaluator = Evaluator()
scorer = RuleScorer()
QualityScorer()
loader = DatasetLoader("examples/sample_data.jsonl")
ConsoleReporter()
ModelComparator()
HtmlReporter()
print("[OK] All classes instantiated")

# Load dataset
samples = loader.load()
assert len(samples) == 3
print(f"[OK] Dataset loaded: {len(samples)} samples")

# Rule scoring
async def t():
    r = await scorer.score("test", "good answer with details and suggestions", "reference: good answer, details, suggestions, summary")
    assert 0 <= r.total_score <= 1
    print(f"[OK] RuleScorer score={r.total_score:.4f} details={r.details}")

asyncio.run(t())

# Cost calc
c = calculate_cost_cny("deepseek-chat", 100, 200)
print(f"[OK] cost_cny(deepseek-chat, 100+200) = {c:.6f}")

# HTML reporter with mock data
mock = [
    EvaluationResult("1", "question?", "answer", 1.0, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}, "deepseek-chat", {"total_score": 0.8}, 0.001),
    EvaluationResult("1", "question?", "answer2", 0.5, {"prompt_tokens": 8, "completion_tokens": 15, "total_tokens": 23}, "qwen-turbo", {"total_score": 0.6}, 0.0003),
]
path = HtmlReporter().generate_report(mock, "verify")
print(f"[OK] HTML report: {path}")

print("\n=== ALL CHECKS PASSED ===")

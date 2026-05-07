"""Core functionality tests for llm-eval-kit."""

import pytest
import json
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_eval_kit.dataset.loader import DatasetLoader, EvaluationSample
from llm_eval_kit.scorers.rule_scorer import RuleScorer
from llm_eval_kit.scorers.llm_judge import LLMJudgeScorer
from llm_eval_kit.utils.cost_calc import (
    calculate_cost_cny, calculate_cost_usd, format_cost,
    CNY_PRICING, USD_PRICING,
)
from llm_eval_kit.utils.config import EvalConfigLoader, EvalConfig, ModelConfig
from llm_eval_kit.reporter.models import EvaluationResult
from llm_eval_kit.reporter.comparator import ModelComparator, ComparisonResult, ModelComparison
from llm_eval_kit.reporter.regression import RegressionReporter, RegressionResult


class TestDatasetLoader:
    def test_load_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write('{"id": "001", "question": "退款多久到账？", "reference": "3-5个工作日"}\n')
            f.write('{"id": "002", "question": "怎么改密码？", "reference": "在设置页面修改"}\n')
            tmp = f.name
        try:
            loader = DatasetLoader(tmp)
            samples = loader.load()
            assert len(samples) == 2
            assert samples[0].id == "001"
            assert samples[0].question == "退款多久到账？"
            assert samples[0].reference == "3-5个工作日"
            # Test validation (returns None, should not raise)
            loader.validate()
        finally:
            os.unlink(tmp)

    def test_load_missing_question(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            f.write('{"id": "001"}\n')
            tmp = f.name
        try:
            loader = DatasetLoader(tmp)
            with pytest.raises(ValueError, match="question"):
                loader.load()
        finally:
            os.unlink(tmp)

    def test_max_samples(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for i in range(10):
                f.write(f'{{"id": "{i:03d}", "question": "Q{i}"}}\n')
            tmp = f.name
        try:
            loader = DatasetLoader(tmp)
            samples = loader.load(max_samples=3)
            assert len(samples) == 3
        finally:
            os.unlink(tmp)


class TestRuleScorer:
    @pytest.mark.asyncio
    async def test_perfect_response(self):
        scorer = RuleScorer()
        result = await scorer.score(
            "退款多久到账？",
            "退款一般在3-5个工作日内到账。建议联系客服。总结：如超时请联系。",
            "3-5个工作日",
            ["退款", "到账"],
        )
        assert 0.0 <= result.total_score <= 1.0
        assert "keyword_match" in result.details

    @pytest.mark.asyncio
    async def test_empty_response(self):
        scorer = RuleScorer()
        result = await scorer.score("test", "", "reference")
        assert result.total_score == 0.0

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(self):
        with pytest.raises(ValueError):
            RuleScorer(keyword_weight=1.0, length_weight=1.0,
                       structure_weight=1.0, completeness_weight=1.0)

    @pytest.mark.asyncio
    async def test_keyword_score(self):
        scorer = RuleScorer()
        # All keywords matched
        r1 = await scorer.score("q", "退款到账问题", "ref", ["退款", "到账"])
        d1 = r1.details["keyword_match"]
        assert d1 > 0

        # No keywords matched
        r2 = await scorer.score("q", "你好", "ref", ["退款", "到账"])
        d2 = r2.details["keyword_match"]
        assert d2 == 0.0

    @pytest.mark.asyncio
    async def test_length_score(self):
        scorer = RuleScorer()
        r1 = await scorer.score("q", "短", "参考回答内容")
        assert r1.details["length_appropriateness"] == 0.0

        r2 = await scorer.score("q", "适" * 100, "参" * 100)
        assert r2.details["length_appropriateness"] > 0

    @pytest.mark.asyncio
    async def test_structure_score(self):
        scorer = RuleScorer()
        r1 = await scorer.score("q", "无结构纯文本", "ref")
        r2 = await scorer.score("q", "# 标题\n1. 列表\n- 项目\n\n第二段", "ref")
        assert r2.details["structure_quality"] > r1.details["structure_quality"]


class TestQualityScorer:
    @pytest.mark.asyncio
    async def test_quality_scorer_alias(self):
        from llm_eval_kit.scorers.rule_scorer import QualityScorer
        scorer = QualityScorer()
        result = await scorer.score("问题", "回答内容", "参考")
        assert 0.0 <= result.total_score <= 1.0


class TestCostCalc:
    def test_calculate_cost_cny(self):
        cost = calculate_cost_cny("qwen-turbo", 100, 200)
        assert cost > 0
        assert cost < 1  # Should be tiny

    def test_calculate_cost_usd(self):
        cost = calculate_cost_usd("gpt-4", 100, 200)
        assert cost > 0
        assert cost < 1

    def test_fuzzy_matching(self):
        cost = calculate_cost_cny("deepseek-chat-v2", 100, 200)
        assert cost > 0

    def test_unknown_model_default(self):
        cost = calculate_cost_cny("unknown-model", 100, 200)
        assert cost > 0  # Should use default pricing

    def test_format_cost_tiny(self):
        result = format_cost(0.003)
        assert "分" in result

    def test_format_cost_normal(self):
        result = format_cost(0.0510)
        assert "¥" in result or "CNY" in result


class TestConfig:
    def test_from_dict_basic(self):
        config = EvalConfigLoader.from_dict({
            "evaluation": {
                "name": "test",
                "data": "data.jsonl",
                "models": [
                    {"model": "deepseek-chat", "api_key": "sk-test", "base_url": "https://test.com/v1"}
                ],
            }
        })
        assert config.name == "test"
        assert config.data_path == "data.jsonl"
        assert len(config.models) == 1
        assert config.models[0].model == "deepseek-chat"

    def test_env_var_substitution(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-env-test")
        config = EvalConfigLoader.from_dict({
            "models": [
                {"model": "test", "api_key": "${TEST_API_KEY}", "base_url": "https://test.com/v1"}
            ],
            "data": "data.jsonl",
        })
        assert config.models[0].api_key == "sk-env-test"

    def test_env_var_missing(self):
        with pytest.raises(ValueError, match="Environment variable"):
            EvalConfigLoader.from_dict({
                "models": [
                    {"model": "test", "api_key": "${MISSING_VAR}", "base_url": "https://test.com/v1"}
                ],
                "data": "data.jsonl",
            })

    def test_to_model_configs(self):
        config = EvalConfigLoader.from_dict({
            "models": [
                {"model": "m1", "api_key": "k1", "base_url": "https://a.com/v1"},
            ],
            "data": "data.jsonl",
        })
        mc = config.to_model_configs()
        assert len(mc) == 1
        assert mc[0]["model"] == "m1"
        assert mc[0]["api_key"] == "k1"


class TestModelComparator:
    def test_compare_models(self):
        comparator = ModelComparator()
        results = [
            EvaluationResult(
                sample_id="1", question="q1", response="r1",
                latency=1.0, token_usage={"total_tokens": 100, "prompt_tokens": 50, "completion_tokens": 50},
                model="model-a", scoring_result={"total_score": 0.9},
            ),
            EvaluationResult(
                sample_id="2", question="q2", response="r2",
                latency=2.0, token_usage={"total_tokens": 200, "prompt_tokens": 100, "completion_tokens": 100},
                model="model-b", scoring_result={"total_score": 0.5},
            ),
        ]
        cr = comparator.compare_models(results)
        assert cr.best_overall_model == "model-a"
        assert len(cr.model_comparisons) == 2


class TestRegressionReporter:
    def test_compare_same(self, tmp_path):
        data = {
            "models": [
                {"name": "m1", "avg_score": 0.8, "avg_latency": 1.0, "total_cost": 0.01, "score_per_cost": 80},
            ]
        }
        p1 = tmp_path / "b.json"
        p2 = tmp_path / "c.json"
        p1.write_text(json.dumps(data), encoding="utf-8")
        p2.write_text(json.dumps(data), encoding="utf-8")

        diff = RegressionReporter.compare(str(p1), str(p2))
        assert diff.total_regressions == 0
        assert diff.total_improvements == 0
        assert "m1" in diff.model_diffs
        assert diff.model_diffs["m1"].score_change == 0.0

    def test_compare_improved(self, tmp_path):
        b = {"models": [{"name": "m1", "avg_score": 0.5, "avg_latency": 1.0, "total_cost": 0.01, "score_per_cost": 50}]}
        c = {"models": [{"name": "m1", "avg_score": 0.9, "avg_latency": 1.0, "total_cost": 0.01, "score_per_cost": 90}]}
        pb = tmp_path / "b.json"
        pc = tmp_path / "c.json"
        pb.write_text(json.dumps(b), encoding="utf-8")
        pc.write_text(json.dumps(c), encoding="utf-8")

        diff = RegressionReporter.compare(str(pb), str(pc))
        assert diff.total_improvements == 1
        assert pytest.approx(diff.model_diffs["m1"].score_change, 0.01) == 0.4

    def test_compare_regression(self, tmp_path):
        b = {"models": [{"name": "m1", "avg_score": 0.9, "avg_latency": 1.0, "total_cost": 0.01, "score_per_cost": 90}]}
        c = {"models": [{"name": "m1", "avg_score": 0.4, "avg_latency": 1.0, "total_cost": 0.01, "score_per_cost": 40}]}
        pb = tmp_path / "b.json"
        pc = tmp_path / "c.json"
        pb.write_text(json.dumps(b), encoding="utf-8")
        pc.write_text(json.dumps(c), encoding="utf-8")

        diff = RegressionReporter.compare(str(pb), str(pc))
        assert diff.total_regressions == 1


class TestRegressionHTML:
    def test_save_diff_html(self, tmp_path):
        data = {
            "models": [
                {"name": "m1", "avg_score": 0.8, "avg_latency": 1.0, "total_cost": 0.01, "score_per_cost": 80},
            ]
        }
        p1 = tmp_path / "b.json"
        p2 = tmp_path / "c.json"
        p1.write_text(json.dumps(data), encoding="utf-8")
        p2.write_text(json.dumps(data), encoding="utf-8")
        diff = RegressionReporter.compare(str(p1), str(p2))

        out = tmp_path / "report.html"
        RegressionReporter.save_diff_html(diff, str(out))
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "LLM 评测回归分析" in html

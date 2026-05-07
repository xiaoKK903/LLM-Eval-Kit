"""
LLM-as-Judge scoring engine.

Uses an LLM to evaluate response quality across three dimensions:
accuracy, completeness, conciseness. Eliminates position bias by
judging twice with swapped reference order.
"""

import re
import json
from typing import Dict, Any, Optional, List

from .base import BaseScorer, ScoreResult
from ..adapters.openai_compat import OpenAICompatibleAdapter


JUDGE_PROMPT_TEMPLATE = """你是一个专业的中文评测裁判。请评估以下回答的质量。

## 问题
{question}

## 参考答案
{reference}

## 待评估回答
{response}

请从以下三个维度评分（每个维度 1-5 分），并给出总体评分（1-5 分）：

1. **准确性**：回答是否准确，是否包含事实错误或与参考答案矛盾的内容
2. **完整性**：回答是否全面覆盖了问题要点
3. **简洁性**：回答是否简洁明了，没有冗余内容

请严格按照以下 JSON 格式输出，不要输出其他内容：
```json
{{
  "accuracy": <1-5的整数>,
  "completeness": <1-5的整数>,
  "conciseness": <1-5的整数>,
  "overall": <1-5的整数>,
  "reasoning": "<简要评分理由>"
}}
```"""


class LLMJudgeScorer(BaseScorer):
    """LLM-as-Judge scorer with position bias elimination."""

    def __init__(self, judge_config: Dict[str, Any]):
        self.judge_config = judge_config
        self.judge_adapter = OpenAICompatibleAdapter(judge_config)

    def _create_judge_prompt(self, question: str, response: str,
                             reference: str, order: str = "reference_first") -> str:
        if order == "reference_first":
            return JUDGE_PROMPT_TEMPLATE.format(
                question=question, reference=reference, response=response
            )
        return JUDGE_PROMPT_TEMPLATE.format(
            question=question, reference=response, response=reference
        )

    def _parse_judge_response(self, text: str) -> Optional[Dict[str, Any]]:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if not json_match:
            return None
        try:
            data = json.loads(json_match.group())
            required = ["accuracy", "completeness", "conciseness", "overall", "reasoning"]
            for key in required:
                if key not in data:
                    return None
            for score_key in ["accuracy", "completeness", "conciseness", "overall"]:
                data[score_key] = int(data[score_key])
                if not (1 <= data[score_key] <= 5):
                    return None
            return data
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    async def _get_judge_score(self, question: str, response: str,
                                reference: str, order: str) -> Optional[Dict[str, Any]]:
        try:
            prompt = self._create_judge_prompt(question, response, reference, order)
            judge_response = await self.judge_adapter.generate(prompt, max_tokens=512, temperature=0.1)
            return self._parse_judge_response(judge_response.text)
        except Exception:
            return None

    def _normalize_score(self, score_1_5: float) -> float:
        return (score_1_5 - 1) / 4.0

    async def score(self, question: str, response: str,
                    reference: Optional[str] = None,
                    expected_keywords: Optional[List[str]] = None) -> ScoreResult:
        reference = reference or question

        score1 = await self._get_judge_score(question, response, reference, "reference_first")
        score2 = await self._get_judge_score(question, response, reference, "response_first")

        scores_available = []
        if score1:
            scores_available.append(score1)
        if score2:
            scores_available.append(score2)

        if not scores_available:
            return ScoreResult(
                total_score=0.0,
                details={"accuracy": 0.0, "completeness": 0.0, "conciseness": 0.0},
                reasoning="LLM-Judge failed to produce valid scores"
            )

        avg = {}
        for key in ["accuracy", "completeness", "conciseness", "overall"]:
            vals = [s[key] for s in scores_available]
            avg[key] = sum(vals) / len(vals)

        score_1_5 = avg.get("overall", sum(avg[k] for k in ["accuracy", "completeness", "conciseness"]) / 3)
        total = self._normalize_score(score_1_5)

        details = {
            "accuracy": self._normalize_score(avg["accuracy"]),
            "completeness": self._normalize_score(avg["completeness"]),
            "conciseness": self._normalize_score(avg["conciseness"]),
        }

        reasoning = (
            f"准确性: {details['accuracy']:.2f} | "
            f"完整性: {details['completeness']:.2f} | "
            f"简洁性: {details['conciseness']:.2f} | "
            f"评判次数: {len(scores_available)}"
        )

        return ScoreResult(
            total_score=round(total, 4),
            details=details,
            reasoning=reasoning,
        )

    async def close(self):
        await self.judge_adapter.close()

"""
Rule-based scoring for LLM responses.

Evaluates responses on: keyword match, length appropriateness,
structure quality, relevance, completeness.
All dimensions are Chinese-business-scenario optimized.
"""

import re
import jieba
from typing import Dict, Any, Optional, List

from .base import BaseScorer, ScoreResult


class RuleScorer(BaseScorer):
    """Multi-dimension rule scorer for Chinese business scenarios."""

    def __init__(self, keyword_weight: float = 0.4, length_weight: float = 0.1,
                 structure_weight: float = 0.2, completeness_weight: float = 0.3):
        total = keyword_weight + length_weight + structure_weight + completeness_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        self.keyword_weight = keyword_weight
        self.length_weight = length_weight
        self.structure_weight = structure_weight
        self.completeness_weight = completeness_weight

    # ── public API ──────────────────────────────────────────

    async def score(self, question: str, response: str, reference: Optional[str] = None,
                    expected_keywords: Optional[List[str]] = None) -> ScoreResult:
        scores = {
            "keyword_score": self._calc_keyword_score(response, reference or "", expected_keywords),
            "length_score": self._calc_length_score(response),
            "structure_score": self._calc_structure_score(response),
            "completeness_score": self._calc_completeness_score(question, response),
        }

        total = (scores["keyword_score"] * self.keyword_weight +
                 scores["length_score"] * self.length_weight +
                 scores["structure_score"] * self.structure_weight +
                 scores["completeness_score"] * self.completeness_weight)

        details = {
            "keyword_match": round(scores["keyword_score"], 4),
            "length_appropriateness": round(scores["length_score"], 4),
            "structure_quality": round(scores["structure_score"], 4),
            "completeness": round(scores["completeness_score"], 4),
        }

        reasoning = (
            f"关键词匹配: {details['keyword_match']:.2f} | "
            f"长度: {details['length_appropriateness']:.2f} | "
            f"结构: {details['structure_quality']:.2f} | "
            f"完整度: {details['completeness']:.2f}"
        )

        return ScoreResult(total_score=round(total, 4), details=details, reasoning=reasoning)

    # ── keyword matching ─────────────────────────────────────

    def _extract_keywords(self, text: str) -> list:
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '他', '她', '它', '我们', '你们', '他们', '这个', '那个',
            '吗', '吧', '啊', '呢', '哦', '嗯', '哈', '呀', '嘛', '么',
        }
        words = jieba.cut(text)
        keywords = []
        for word in words:
            word = word.strip()
            if (len(word) >= 2 and word not in stop_words and not word.isdigit()
                    and not all(c in '，。！？；：""''（）【】《》' for c in word)):
                keywords.append(word)
        return keywords

    def _calc_keyword_score(self, response: str, reference: str,
                            expected_keywords: Optional[List[str]] = None) -> float:
        if expected_keywords:
            target = expected_keywords
        else:
            target = self._extract_keywords(reference)

        if not target:
            return 0.5

        resp_keywords = self._extract_keywords(response)
        matched = sum(1 for k in target if k in resp_keywords)
        return min(1.0, matched / len(target))

    # ── length appropriateness ──────────────────────────────

    def _calc_length_score(self, response: str) -> float:
        length = len(response)
        if length < 20:
            return 0.0
        if length < 100:
            return length / 100
        if length <= 500:
            return 1.0
        if length <= 1000:
            return 1.0 - (length - 500) / 500 * 0.3
        return max(0.2, 1.0 - (length - 1000) / 1000 * 0.5)

    # ── structure quality ──────────────────────────────────

    def _calc_structure_score(self, response: str) -> float:
        score = 0.0
        if re.search(r'(^|\n)#{1,3}\s', response):
            score += 0.3
        if re.search(r'\d+\.\s', response):
            score += 0.2
        if re.search(r'[-*]\s', response):
            score += 0.2
        if re.search(r'\*\*.*?\*\*', response):
            score += 0.1
        paragraphs = [p for p in response.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 0.2
        return min(1.0, score)

    # ── completeness (Chinese business keywords) ─────────────

    def _calc_completeness_score(self, question: str, response: str) -> float:
        indicators = [
            ('总结', 0.15), ('建议', 0.15), ('步骤', 0.15), ('原因', 0.15),
            ('注意', 0.1), ('方案', 0.1), ('说明', 0.1), ('例如', 0.1),
        ]
        score = 0.0
        resp_lower = response.lower()
        for indicator, weight in indicators:
            if indicator in resp_lower:
                score += weight
        if len(response) > 300 and score > 0.3:
            score += 0.1
        return min(1.0, score)


class QualityScorer(RuleScorer):
    """Legacy alias for RuleScorer with equal-weight defaults."""
    def __init__(self):
        super().__init__(keyword_weight=0.25, length_weight=0.25,
                         structure_weight=0.25, completeness_weight=0.25)

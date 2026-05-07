"""
Base classes for LLM response scoring.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Result of scoring a model response."""

    total_score: float
    details: Dict[str, Any]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_score": self.total_score,
            "details": self.details,
            "reasoning": self.reasoning,
        }


class BaseScorer:
    """Base class for all scoring implementations."""

    async def score(self, question: str, response: str,
                    reference: Optional[str] = None,
                    expected_keywords: Optional[list] = None) -> ScoreResult:
        raise NotImplementedError

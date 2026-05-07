"""
Base classes for LLM response scoring.

This module defines the base interface for all scoring implementations.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Result of scoring a model response."""
    
    total_score: float
    """Overall score in range 0-1."""
    
    details: Dict[str, Any]
    """Detailed scoring breakdown."""
    
    reasoning: str
    """Explanation of the scoring decision."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_score": self.total_score,
            "details": self.details,
            "reasoning": self.reasoning
        }


class BaseScorer:
    """Base class for all scoring implementations."""
    
    def score(self, question: str, response: str, reference: str) -> ScoreResult:
        """
        Score a model response against a reference answer.
        
        Args:
            question: The original question asked
            response: The model's response
            reference: The reference (correct) answer
            
        Returns:
            ScoreResult containing the score and reasoning
            
        Raises:
            Exception: If scoring fails
        """
        raise NotImplementedError("Subclasses must implement score method")
    
    def batch_score(self, samples: list) -> list:
        """
        Score multiple samples in batch.
        
        Args:
            samples: List of dictionaries with 'question', 'response', 'reference'
            
        Returns:
            List of ScoreResult objects
        """
        results = []
        for sample in samples:
            result = self.score(
                sample["question"],
                sample["response"],
                sample["reference"]
            )
            results.append(result)
        return results
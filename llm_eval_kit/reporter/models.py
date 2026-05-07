"""
Data models for evaluation results.

This module contains the data classes used to represent evaluation results,
separated from the reporting logic to avoid circular imports.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Represents the result of evaluating a single sample."""
    
    sample_id: str
    """Unique identifier for the sample."""
    
    question: str
    """The original question asked."""
    
    response: str
    """The model's response."""
    
    latency: float
    """Response latency in seconds."""
    
    token_usage: Dict[str, int]
    """Token usage statistics."""
    
    model: str
    """Name of the model that generated the response."""
    
    quality_scores: Optional[Dict[str, float]] = None
    """Quality scores from quality scorer."""
    
    cost: float = 0.0
    """Cost of the response in USD."""
    
    scoring_result: Optional[Dict[str, Any]] = None
    """Scoring result from scoring engine."""
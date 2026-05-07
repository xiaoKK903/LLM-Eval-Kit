"""
Model comparison reporter for LLM evaluation results.

This module provides functionality to compare multiple models based on
evaluation results and generate comprehensive comparison reports.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..reporter.models import EvaluationResult


@dataclass
class ModelComparison:
    """Comparison data for a single model."""
    
    model_name: str
    """Name of the model."""
    
    avg_score: float
    """Average scoring result (0-1)."""
    
    avg_latency: float
    """Average latency in seconds."""
    
    total_tokens: int
    """Total tokens used."""
    
    total_cost: float
    """Total cost in Chinese Yuan (¥)."""
    
    success_rate: float
    """Success rate (0-1)."""
    
    sample_count: int
    """Number of samples evaluated."""
    
    cost_per_sample: float
    """Average cost per sample in ¥."""
    
    score_per_cost: float
    """Score per cost unit (score / cost)."""


@dataclass
class ComparisonResult:
    """Complete comparison results for multiple models."""
    
    model_comparisons: List[ModelComparison]
    """Comparison data for each model."""
    
    best_overall_model: str
    """Model with highest average score."""
    
    fastest_model: str
    """Model with lowest average latency."""
    
    cheapest_model: str
    """Model with lowest total cost."""
    
    best_value_model: str
    """Model with highest score per cost ratio."""
    
    best_overall_score: float
    """Score of the best overall model."""
    
    fastest_latency: float
    """Latency of the fastest model."""
    
    cheapest_cost: float
    """Cost of the cheapest model."""
    
    best_value_ratio: float
    """Score per cost ratio of the best value model."""


class ModelComparator:
    """Comparator for comparing multiple LLM models based on evaluation results."""
    
    # Default pricing in Chinese Yuan per million tokens (¥/百万token)
    # Only includes models we have actually tested
    DEFAULT_PRICING = {
        # Alibaba Qwen series
        "qwen-turbo": {"input": 0.3, "output": 0.6},
        "qwen-plus": {"input": 0.8, "output": 2.0},
        "qwen-max": {"input": 2.0, "output": 4.0},

        # DeepSeek series
        "deepseek-chat": {"input": 1.0, "output": 2.0},
        "deepseek-coder": {"input": 1.0, "output": 2.0},

        # OpenAI series
        "gpt-3.5-turbo": {"input": 1.5, "output": 2.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 15.0, "output": 30.0},

        # Default fallback pricing
        "default": {"input": 1.0, "output": 2.0}
    }
    
    def __init__(self, custom_pricing: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the model comparator.
        
        Args:
            custom_pricing: Custom pricing configuration for models
        """
        self.pricing = self.DEFAULT_PRICING.copy()
        if custom_pricing:
            self.pricing.update(custom_pricing)
    
    def _calculate_cost(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate cost for a model based on token usage.
        
        Args:
            model_name: Name of the model
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            
        Returns:
            Cost in Chinese Yuan (¥)
        """
        # Get pricing for the model, fallback to default if not found
        model_pricing = self.pricing.get(model_name, self.pricing.get("default", {"input": 1.0, "output": 2.0}))
        
        # Convert per million tokens to per token
        input_price_per_token = model_pricing["input"] / 1_000_000
        output_price_per_token = model_pricing["output"] / 1_000_000
        
        # Calculate cost
        input_cost = prompt_tokens * input_price_per_token
        output_cost = completion_tokens * output_price_per_token
        
        return input_cost + output_cost
    
    def _group_results_by_model(self, results: List[EvaluationResult]) -> Dict[str, List[EvaluationResult]]:
        """
        Group evaluation results by model name.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary mapping model names to their results
        """
        grouped = {}
        for result in results:
            model_name = result.model
            if model_name not in grouped:
                grouped[model_name] = []
            grouped[model_name].append(result)
        return grouped
    
    def _calculate_model_stats(self, model_name: str, results: List[EvaluationResult]) -> ModelComparison:
        """
        Calculate statistics for a single model.
        
        Args:
            model_name: Name of the model
            results: Evaluation results for this model
            
        Returns:
            ModelComparison object with calculated statistics
        """
        sample_count = len(results)
        
        # Calculate average score
        total_score = 0
        scored_samples = 0
        for result in results:
            if result.scoring_result:
                total_score += result.scoring_result.get("total_score", 0)
                scored_samples += 1
        
        avg_score = total_score / scored_samples if scored_samples > 0 else 0
        
        # Calculate average latency
        total_latency = sum(result.latency for result in results)
        avg_latency = total_latency / sample_count if sample_count > 0 else 0
        
        # Calculate total tokens
        total_prompt_tokens = sum(result.token_usage.get("prompt_tokens", 0) for result in results)
        total_completion_tokens = sum(result.token_usage.get("completion_tokens", 0) for result in results)
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        # Calculate total cost
        total_cost = self._calculate_cost(model_name, total_prompt_tokens, total_completion_tokens)
        
        # Calculate success rate (assuming all results are successful for now)
        success_rate = 1.0  # We'll improve this later with error handling
        
        # Calculate derived metrics
        cost_per_sample = total_cost / sample_count if sample_count > 0 else 0
        score_per_cost = avg_score / cost_per_sample if cost_per_sample > 0 else 0
        
        return ModelComparison(
            model_name=model_name,
            avg_score=avg_score,
            avg_latency=avg_latency,
            total_tokens=total_tokens,
            total_cost=total_cost,
            success_rate=success_rate,
            sample_count=sample_count,
            cost_per_sample=cost_per_sample,
            score_per_cost=score_per_cost
        )
    
    def _determine_best_models(self, comparisons: List[ModelComparison]) -> Dict[str, Any]:
        """
        Determine the best models in different categories.
        
        Args:
            comparisons: List of model comparisons
            
        Returns:
            Dictionary with best model information
        """
        if not comparisons:
            return {
                "best_overall_model": "N/A",
                "fastest_model": "N/A", 
                "cheapest_model": "N/A",
                "best_value_model": "N/A",
                "best_overall_score": 0,
                "fastest_latency": 0,
                "cheapest_cost": 0,
                "best_value_ratio": 0
            }
        
        # Find best overall (highest score)
        best_overall = max(comparisons, key=lambda x: x.avg_score)
        
        # Find fastest (lowest latency)
        fastest = min(comparisons, key=lambda x: x.avg_latency)
        
        # Find cheapest (lowest cost per sample)
        cheapest = min(comparisons, key=lambda x: x.cost_per_sample)
        
        # Find best value (highest score per cost)
        best_value = max(comparisons, key=lambda x: x.score_per_cost)
        
        return {
            "best_overall_model": best_overall.model_name,
            "fastest_model": fastest.model_name,
            "cheapest_model": cheapest.model_name,
            "best_value_model": best_value.model_name,
            "best_overall_score": best_overall.avg_score,
            "fastest_latency": fastest.avg_latency,
            "cheapest_cost": cheapest.cost_per_sample,
            "best_value_ratio": best_value.score_per_cost
        }
    
    def compare_models(self, results: List[EvaluationResult]) -> ComparisonResult:
        """
        Compare multiple models based on evaluation results.
        
        Args:
            results: List of evaluation results from multiple models
            
        Returns:
            ComparisonResult with comprehensive comparison data
        """
        # Group results by model
        grouped_results = self._group_results_by_model(results)
        
        # Calculate statistics for each model
        model_comparisons = []
        for model_name, model_results in grouped_results.items():
            comparison = self._calculate_model_stats(model_name, model_results)
            model_comparisons.append(comparison)
        
        # Determine best models
        best_models = self._determine_best_models(model_comparisons)
        
        return ComparisonResult(
            model_comparisons=model_comparisons,
            **best_models
        )
    
    def get_pricing_info(self, model_name: str) -> Dict[str, float]:
        """
        Get pricing information for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with input and output pricing
        """
        return self.pricing.get(model_name, {"input": 1.0, "output": 2.0})
    
    def add_custom_pricing(self, model_name: str, input_price: float, output_price: float):
        """
        Add custom pricing for a model.
        
        Args:
            model_name: Name of the model
            input_price: Price per million input tokens in ¥
            output_price: Price per million output tokens in ¥
        """
        self.pricing[model_name] = {
            "input": input_price,
            "output": output_price
        }
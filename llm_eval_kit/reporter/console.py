"""
Console reporter for displaying evaluation results.

This module provides functionality to display evaluation results in a formatted
table format in the console.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Represents the result of evaluating a single sample."""
    
    sample_id: str
    question: str
    response: str
    latency: float
    token_usage: Dict[str, int]
    model: str
    quality_scores: Dict[str, float] = None
    cost: float = 0.0
    scoring_result: Optional[Dict[str, Any]] = None


class ConsoleReporter:
    """Reporter for displaying evaluation results in the console."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the console reporter.
        
        Args:
            verbose: Whether to display detailed information (default: False)
        """
        self.verbose = verbose
    
    def print_results(self, results: List[EvaluationResult]) -> None:
        """
        Print evaluation results in a formatted table.
        
        Args:
            results: List of evaluation results to display
        """
        if not results:
            print("No results to display.")
            return
        
        print("\n" + "="*80)
        print("LLM Evaluation Results")
        print("="*80)
        
        # Print summary statistics
        self._print_summary(results)
        
        # Print detailed results if verbose mode is enabled
        if self.verbose:
            self._print_detailed_results(results)
        
        print("="*80)
    
    def _print_summary(self, results: List[EvaluationResult]) -> None:
        """Print summary statistics."""
        total_samples = len(results)
        total_latency = sum(result.latency for result in results)
        avg_latency = total_latency / total_samples if total_samples > 0 else 0
        
        total_tokens = sum(result.token_usage.get("total_tokens", 0) for result in results)
        total_prompt_tokens = sum(result.token_usage.get("prompt_tokens", 0) for result in results)
        total_completion_tokens = sum(result.token_usage.get("completion_tokens", 0) for result in results)
        
        # Calculate quality scores and cost
        total_cost = sum(result.cost for result in results)
        avg_cost = total_cost / total_samples if total_samples > 0 else 0
        
        # Calculate average quality scores
        quality_metrics = {}
        if results and results[0].quality_scores:
            for metric in results[0].quality_scores.keys():
                scores = [r.quality_scores.get(metric, 0) for r in results if r.quality_scores]
                if scores:
                    quality_metrics[metric] = sum(scores) / len(scores)
        
        # Calculate scoring metrics if available
        scoring_metrics = {}
        if results and results[0].scoring_result:
            scoring_results = [r.scoring_result for r in results if r.scoring_result]
            if scoring_results:
                total_scoring_score = sum(r.get("total_score", 0) for r in scoring_results)
                avg_scoring_score = total_scoring_score / len(scoring_results)
                scoring_metrics["average_score"] = avg_scoring_score
        
        print(f"\nSummary:")
        print(f"  Total Samples: {total_samples}")
        print(f"  Average Latency: {avg_latency:.2f}s")
        print(f"  Total Tokens: {total_tokens}")
        print(f"    - Prompt Tokens: {total_prompt_tokens}")
        print(f"    - Completion Tokens: {total_completion_tokens}")
        
        # Show quality scores
        if quality_metrics:
            print(f"  Quality Scores:")
            for metric, score in quality_metrics.items():
                print(f"    - {metric}: {score:.2f}")
        
        # Show scoring results
        if scoring_metrics:
            print(f"  Scoring Results:")
            print(f"    - Average Score: {scoring_metrics['average_score']:.3f}")
        
        # Show cost information
        print(f"  Total Cost: ${total_cost:.4f}")
        print(f"  Average Cost per Sample: ${avg_cost:.4f}")
        
        # Show model information
        models = set(result.model for result in results)
        if len(models) == 1:
            print(f"  Model: {next(iter(models))}")
        else:
            print(f"  Models: {', '.join(models)}")
    
    def _print_detailed_results(self, results: List[EvaluationResult]) -> None:
        """Print detailed results for each sample."""
        print(f"\nDetailed Results ({len(results)} samples):")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"\nSample {i} (ID: {result.sample_id}):")
            print(f"  Question: {self._truncate_text(result.question, 60)}")
            print(f"  Response: {self._truncate_text(result.response, 60)}")
            print(f"  Latency: {result.latency:.2f}s")
            print(f"  Tokens: {result.token_usage.get('total_tokens', 0)} "
                  f"(P: {result.token_usage.get('prompt_tokens', 0)}, "
                  f"C: {result.token_usage.get('completion_tokens', 0)})")
            print(f"  Model: {result.model}")
            
            # Print scoring results if available
            if result.scoring_result:
                score = result.scoring_result.get("total_score", 0)
                reasoning = result.scoring_result.get("reasoning", "")
                print(f"  Score: {score:.3f}")
                print(f"  Reasoning: {self._truncate_text(reasoning, 50)}")
            
            # Print full text if it was truncated
            if len(result.question) > 60:
                print(f"  Full Question: {result.question}")
            if len(result.response) > 60:
                print(f"  Full Response: {result.response}")
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """
        Truncate text to specified maximum length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length of the truncated text
            
        Returns:
            Truncated text with ellipsis if necessary
        """
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    def print_single_result(self, result: EvaluationResult) -> None:
        """
        Print a single evaluation result.
        
        Args:
            result: The evaluation result to display
        """
        print("\n" + "-"*40)
        print("Single Evaluation Result")
        print("-"*40)
        
        print(f"ID: {result.sample_id}")
        print(f"Question: {result.question}")
        print(f"Response: {result.response}")
        print(f"Latency: {result.latency:.2f}s")
        print(f"Tokens: {result.token_usage.get('total_tokens', 0)}")
        print(f"Model: {result.model}")
        
        # Print detailed scoring information
        if result.scoring_result:
            print(f"\nScoring Details:")
            print(f"  Total Score: {result.scoring_result.get('total_score', 0):.3f}")
            print(f"  Reasoning: {result.scoring_result.get('reasoning', '')}")
            
            # Print detailed breakdown if available
            details = result.scoring_result.get('details', {})
            if details:
                print(f"  Details:")
                for key, value in details.items():
                    if isinstance(value, float):
                        print(f"    - {key}: {value:.3f}")
                    else:
                        print(f"    - {key}: {value}")
        
        print("-"*40)
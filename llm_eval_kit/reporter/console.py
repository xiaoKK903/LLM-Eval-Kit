"""
Console reporter for displaying evaluation results.

This module provides functionality to display evaluation results in a formatted
table format in the console.
"""

from typing import List, Dict, Any
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
        
        print(f"\nSummary:")
        print(f"  Total Samples: {total_samples}")
        print(f"  Average Latency: {avg_latency:.2f}s")
        print(f"  Total Tokens: {total_tokens}")
        print(f"    - Prompt Tokens: {total_prompt_tokens}")
        print(f"    - Completion Tokens: {total_completion_tokens}")
        
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
        print("-"*40)
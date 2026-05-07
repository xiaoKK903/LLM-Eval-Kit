"""
Main client for LLM evaluation with concurrent execution support.

This module provides the main interface for running LLM evaluations,
coordinating between dataset loading, model inference, and result reporting.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from asyncio import Semaphore

from .dataset.loader import DatasetLoader, EvaluationSample
from .adapters.openai_compatible import OpenAICompatibleHttpxAdapter
from .reporter.console import ConsoleReporter
from .reporter.models import EvaluationResult
from .metrics.quality_scorer import QualityScorer, calculate_cost


@dataclass
class EvaluationConfig:
    """Configuration for running an evaluation."""
    
    data_path: str
    """Path to the dataset file (JSONL format)."""
    
    model_config: Dict[str, Any]
    """Configuration for the model adapter."""
    
    verbose: bool = False
    """Whether to display detailed output."""
    
    max_samples: Optional[int] = None
    """Maximum number of samples to evaluate (None for all)."""
    
    concurrency_per_model: int = 3
    """Maximum concurrent requests per model."""
    
    model_level_concurrency: int = 4
    """Maximum concurrent models to evaluate."""


class EvalClient:
    """Main client for running LLM evaluations with concurrent execution."""
    
    def __init__(self):
        """Initialize the evaluation client."""
        self.dataset_loader: Optional[DatasetLoader] = None
        self.model_adapter: Optional[OpenAICompatibleAdapter] = None
        self.reporter: Optional[ConsoleReporter] = None
        
        # Statistics
        self.success_count = 0
        self.failure_count = 0
        self.total_latency = 0.0
    
    async def evaluate(self, config: EvaluationConfig) -> List[EvaluationResult]:
        """
        Run a complete evaluation pipeline with concurrent execution.
        
        Args:
            config: Evaluation configuration
            
        Returns:
            List of evaluation results
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        start_time = time.time()
        
        # Initialize components
        self._initialize_components(config)
        
        # Load dataset
        samples = self._load_dataset(config)
        
        # Run evaluation with concurrency control
        results = await self._run_concurrent_evaluation(samples, config)
        
        # Report results
        self._report_results(results, config)
        
        # Print performance summary
        total_time = time.time() - start_time
        self._print_performance_summary(total_time, len(samples))
        
        return results
    
    def _initialize_components(self, config: EvaluationConfig) -> None:
        """Initialize the evaluation components."""
        # Initialize model adapter
        try:
            self.model_adapter = OpenAICompatibleHttpxAdapter(config.model_config)
        except Exception as e:
            raise Exception(f"Failed to initialize model adapter: {e}")
        
        # Initialize reporter
        self.reporter = ConsoleReporter(verbose=config.verbose)
        
        # Reset statistics
        self.success_count = 0
        self.failure_count = 0
        self.total_latency = 0.0
    
    def _load_dataset(self, config: EvaluationConfig) -> List[EvaluationSample]:
        """Load and validate the evaluation dataset."""
        try:
            self.dataset_loader = DatasetLoader(config.data_path)
            samples = self.dataset_loader.load()
            
            # Validate dataset
            self.dataset_loader.validate()
            
            # Apply sample limit if specified
            if config.max_samples is not None:
                samples = samples[:config.max_samples]
            
            print(f"Loaded {len(samples)} samples from {config.data_path}")
            return samples
            
        except Exception as e:
            raise Exception(f"Failed to load dataset: {e}")
    
    async def _run_concurrent_evaluation(self, samples: List[EvaluationSample], 
                                        config: EvaluationConfig) -> List[EvaluationResult]:
        """Run evaluation on all samples with concurrency control."""
        
        print(f"Starting concurrent evaluation of {len(samples)} samples...")
        print(f"Concurrency settings: {config.concurrency_per_model} per model, "
              f"{config.model_level_concurrency} models max")
        
        # Initialize quality scorer
        quality_scorer = QualityScorer()
        
        # Create semaphores for concurrency control
        model_semaphore = Semaphore(config.model_level_concurrency)
        sample_semaphore = Semaphore(config.concurrency_per_model)
        
        # Create tasks for all samples
        tasks = []
        for sample in samples:
            task = self._evaluate_sample_with_semaphores(
                sample, quality_scorer, model_semaphore, sample_semaphore, config.verbose
            )
            tasks.append(task)
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and collect successful results
        successful_results = []
        for result in results:
            if isinstance(result, Exception):
                self.failure_count += 1
                if config.verbose:
                    print(f"Sample evaluation failed: {result}")
            else:
                successful_results.append(result)
                self.success_count += 1
                self.total_latency += result.latency
        
        return successful_results
    
    async def _evaluate_sample_with_semaphores(self, sample: EvaluationSample,
                                              quality_scorer: QualityScorer,
                                              model_semaphore: Semaphore,
                                              sample_semaphore: Semaphore,
                                              verbose: bool) -> EvaluationResult:
        """Evaluate a single sample with concurrency control."""
        
        async with model_semaphore:
            async with sample_semaphore:
                if verbose:
                    print(f"Evaluating sample {sample.id} (concurrent)")
                
                try:
                    # Generate response using the model
                    response = await self.model_adapter.generate(sample.question)
                    
                    # Calculate quality scores
                    quality_scores = quality_scorer.score_response(
                        sample.question, 
                        response.text,
                        sample.expected_keywords if hasattr(sample, 'expected_keywords') else None
                    )
                    
                    # Calculate cost
                    total_tokens = response.token_usage.get('total_tokens', 0)
                    cost = calculate_cost(total_tokens, response.model)
                    
                    # Create evaluation result
                    result = EvaluationResult(
                        sample_id=sample.id,
                        question=sample.question,
                        response=response.text,
                        latency=response.latency,
                        token_usage=response.token_usage,
                        model=response.model,
                        quality_scores=quality_scores,
                        cost=cost
                    )
                    
                    return result
                    
                except Exception as e:
                    # Re-raise the exception to be handled by the caller
                    raise Exception(f"Error evaluating sample {sample.id}: {e}")
    
    async def evaluate_samples(self, samples: List[EvaluationSample], scorer) -> List[EvaluationResult]:
        """
        Evaluate a list of samples directly without loading from file.
        
        Args:
            samples: List of evaluation samples
            scorer: Scoring engine to use
            
        Returns:
            List of evaluation results
        """
        if not samples:
            return []
        
        # Initialize model adapter if not already done
        if not self.model_adapter:
            # Use a default config for the adapter
            default_config = {
                "base_url": "https://api.openai.com/v1",
                "api_key": "dummy",
                "model": "gpt-3.5-turbo",
                "timeout": 30
            }
            self.model_adapter = OpenAICompatibleHttpxAdapter(default_config)
        
        # Initialize reporter
        self.reporter = ConsoleReporter(verbose=False)
        
        # Evaluate each sample
        results = []
        for sample in samples:
            try:
                # Generate response
                response = await self.model_adapter.generate(sample.question)
                
                # Score the response
                scoring_result = scorer.score(sample.question, response.text, sample.reference)
                
                # Create evaluation result
                result = EvaluationResult(
                    sample_id=sample.id,
                    question=sample.question,
                    response=response.text,
                    latency=response.latency,
                    token_usage=response.token_usage,
                    model=self.model_adapter.model_name,
                    scoring_result=scoring_result.to_dict()
                )
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating sample {sample.id}: {e}")
                continue
        
        return results
    
    def _report_results(self, results: List[EvaluationResult], 
                       config: EvaluationConfig) -> None:
        """Generate and display evaluation reports."""
        if not results:
            print("No results to report.")
            return
        
        # Print console report
        self.reporter.print_results(results)
    
    def _print_performance_summary(self, total_time: float, total_samples: int) -> None:
        """Print performance summary after evaluation."""
        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        print(f"Total Evaluation Time: {total_time:.2f}s")
        print(f"Total Samples: {total_samples}")
        print(f"Successful Evaluations: {self.success_count}")
        print(f"Failed Evaluations: {self.failure_count}")
        
        if self.success_count > 0:
            avg_latency = self.total_latency / self.success_count
            print(f"Average Latency: {avg_latency:.2f}s")
            print(f"Samples per Second: {self.success_count / total_time:.2f}")
        
        success_rate = (self.success_count / total_samples) * 100 if total_samples > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")


async def run_evaluation(data_path: str, model_config: Dict[str, Any], 
                        verbose: bool = False, max_samples: Optional[int] = None,
                        concurrency_per_model: int = 3, model_level_concurrency: int = 4) -> List[EvaluationResult]:
    """
    Convenience function to run an evaluation with default settings.
    
    Args:
        data_path: Path to the dataset file
        model_config: Configuration for the model adapter
        verbose: Whether to display detailed output
        max_samples: Maximum number of samples to evaluate
        concurrency_per_model: Maximum concurrent requests per model
        model_level_concurrency: Maximum concurrent models to evaluate
        
    Returns:
        List of evaluation results
    """
    config = EvaluationConfig(
        data_path=data_path,
        model_config=model_config,
        verbose=verbose,
        max_samples=max_samples,
        concurrency_per_model=concurrency_per_model,
        model_level_concurrency=model_level_concurrency
    )
    
    client = EvalClient()
    return await client.evaluate(config)
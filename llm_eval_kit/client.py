"""
Main client for LLM evaluation.

This module provides the main interface for running LLM evaluations,
coordinating between dataset loading, model inference, and result reporting.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .dataset.loader import DatasetLoader, EvaluationSample
from .adapters.openai_compatible import OpenAICompatibleAdapter
from .reporter.console import ConsoleReporter, EvaluationResult


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


class EvalClient:
    """Main client for running LLM evaluations."""
    
    def __init__(self):
        """Initialize the evaluation client."""
        self.dataset_loader: Optional[DatasetLoader] = None
        self.model_adapter: Optional[OpenAICompatibleAdapter] = None
        self.reporter: Optional[ConsoleReporter] = None
    
    async def evaluate(self, config: EvaluationConfig) -> List[EvaluationResult]:
        """
        Run a complete evaluation pipeline.
        
        Args:
            config: Evaluation configuration
            
        Returns:
            List of evaluation results
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        # Initialize components
        self._initialize_components(config)
        
        # Load dataset
        samples = self._load_dataset(config)
        
        # Run evaluation
        results = await self._run_evaluation(samples, config)
        
        # Report results
        self._report_results(results, config)
        
        return results
    
    def _initialize_components(self, config: EvaluationConfig) -> None:
        """Initialize the evaluation components."""
        # Initialize model adapter
        try:
            self.model_adapter = OpenAICompatibleAdapter(config.model_config)
        except Exception as e:
            raise Exception(f"Failed to initialize model adapter: {e}")
        
        # Initialize reporter
        self.reporter = ConsoleReporter(verbose=config.verbose)
    
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
    
    async def _run_evaluation(self, samples: List[EvaluationSample], 
                            config: EvaluationConfig) -> List[EvaluationResult]:
        """Run evaluation on all samples."""
        results = []
        
        print(f"Starting evaluation of {len(samples)} samples...")
        
        for i, sample in enumerate(samples, 1):
            try:
                if config.verbose:
                    print(f"Evaluating sample {i}/{len(samples)} (ID: {sample.id})")
                
                # Generate response using the model
                response = await self.model_adapter.generate(sample.question)
                
                # Create evaluation result
                result = EvaluationResult(
                    sample_id=sample.id,
                    question=sample.question,
                    response=response.text,
                    latency=response.latency,
                    token_usage=response.token_usage,
                    model=response.model
                )
                
                results.append(result)
                
                # Print progress
                if i % 10 == 0 or i == len(samples):
                    print(f"Progress: {i}/{len(samples)} samples completed")
                
            except Exception as e:
                print(f"Error evaluating sample {sample.id}: {e}")
                # Continue with next sample even if one fails
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
        
        # Print summary
        successful_samples = len(results)
        total_samples = len(self.dataset_loader._samples) if self.dataset_loader else 0
        
        print(f"\nEvaluation completed: {successful_samples}/{total_samples} samples successful")
    
    async def evaluate_single(self, question: str, model_config: Dict[str, Any]) -> EvaluationResult:
        """
        Evaluate a single question.
        
        Args:
            question: The question to evaluate
            model_config: Configuration for the model adapter
            
        Returns:
            Evaluation result for the single question
        """
        # Initialize model adapter
        adapter = OpenAICompatibleAdapter(model_config)
        
        # Generate response
        response = await adapter.generate(question)
        
        # Create and return result
        return EvaluationResult(
            sample_id="single",
            question=question,
            response=response.text,
            latency=response.latency,
            token_usage=response.token_usage,
            model=response.model
        )


# Convenience function for simple evaluations
async def run_evaluation(data_path: str, model_config: Dict[str, Any], 
                        verbose: bool = False, max_samples: Optional[int] = None) -> List[EvaluationResult]:
    """
    Convenience function to run an evaluation with minimal configuration.
    
    Args:
        data_path: Path to the dataset file
        model_config: Model configuration
        verbose: Whether to display detailed output
        max_samples: Maximum number of samples to evaluate
        
    Returns:
        List of evaluation results
    """
    client = EvalClient()
    config = EvaluationConfig(
        data_path=data_path,
        model_config=model_config,
        verbose=verbose,
        max_samples=max_samples
    )
    
    return await client.evaluate(config)
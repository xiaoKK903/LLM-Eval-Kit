"""
Console reporter for displaying evaluation results.

This module provides functionality to display evaluation results in a formatted
table format in the console.
"""

from typing import List, Dict, Any, Optional
from .models import EvaluationResult
from .comparator import ModelComparator, ComparisonResult


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

    def _get_display_width(self, text: str) -> int:
        """
        Calculate display width considering Chinese characters.
        
        Args:
            text: Text to calculate width for
            
        Returns:
            Display width (Chinese characters count as 2)
        """
        width = 0
        for char in text:
            # Chinese characters have width 2, others have width 1
            if '\u4e00' <= char <= '\u9fff':
                width += 2
            else:
                width += 1
        return width
    
    def _pad_text(self, text: str, width: int, align: str = 'left') -> str:
        """
        Pad text to specified width considering Chinese characters.
        
        Args:
            text: Text to pad
            width: Target width
            align: Alignment ('left', 'right', 'center')
            
        Returns:
            Padded text
        """
        text_width = self._get_display_width(text)
        
        if text_width >= width:
            return text
        
        padding = width - text_width
        
        if align == 'left':
            return text + ' ' * padding
        elif align == 'right':
            return ' ' * padding + text
        else:  # center
            left_padding = padding // 2
            right_padding = padding - left_padding
            return ' ' * left_padding + text + ' ' * right_padding
    
    def print_comparison_report(self, results: List[EvaluationResult], 
                              custom_pricing: Optional[Dict[str, Dict[str, float]]] = None) -> None:
        """
        Print a comprehensive comparison report for multiple models.
        
        Args:
            results: List of evaluation results from multiple models
            custom_pricing: Custom pricing configuration for models
        """
        if not results:
            print("No results to compare.")
            return
        
        # Create comparator and generate comparison
        comparator = ModelComparator(custom_pricing)
        comparison_result = comparator.compare_models(results)
        
        # Print header
        header_width = 80
        print("\n" + "═" * header_width)
        print(self._pad_text("模型评测对比报告", header_width, 'center'))
        print("═" * header_width)
        
        # Print table header
        column_widths = [14, 8, 8, 10, 10, 8]  # Model, Score, Latency, Token, Cost, Success Rate
        headers = ["模型名称", "平均分", "延迟(s)", "Token", "成本(¥)", "成功率"]
        
        header_line = ""
        for i, header in enumerate(headers):
            header_line += self._pad_text(header, column_widths[i], 'center')
        
        print(header_line)
        print("─" * sum(column_widths))
        
        # Print model data
        for comparison in comparison_result.model_comparisons:
            row_data = [
                comparison.model_name,
                f"{comparison.avg_score:.2f}",
                f"{comparison.avg_latency:.1f}",
                f"{comparison.total_tokens}",
                f"¥{comparison.total_cost:.4f}",
                f"{comparison.success_rate * 100:.0f}%"
            ]
            
            row_line = ""
            for i, data in enumerate(row_data):
                align = 'right' if i > 0 else 'left'  # Model name left-aligned, others right
                row_line += self._pad_text(data, column_widths[i], align)
            
            print(row_line)
        
        print("─" * sum(column_widths))
        
        # Print comparison conclusions
        print("\n" + "🏆 综合最优：" + comparison_result.best_overall_model + 
              f"    （得分最高 {comparison_result.best_overall_score:.2f}）")
        print("⚡ 速度最快：" + comparison_result.fastest_model + 
              f"   （平均 {comparison_result.fastest_latency:.1f}s）")
        print("💰 成本最低：" + comparison_result.cheapest_model + 
              f"   （¥{comparison_result.cheapest_cost:.4f}/次）")
        print("📈 性价比王：" + comparison_result.best_value_model + 
              f"   （得分/成本比最高）")
        
        print("═" * header_width)
        
        # Print pricing information
        print("\n💡 成本计算说明：")
        for comparison in comparison_result.model_comparisons:
            pricing = comparator.get_pricing_info(comparison.model_name)
            print(f"   {comparison.model_name}: " +
                  f"输入 ¥{pricing['input']}/百万token, " +
                  f"输出 ¥{pricing['output']}/百万token")
        
        print("\n📊 评测总结：")
        print(f"   总样本数: {sum(c.sample_count for c in comparison_result.model_comparisons)}")
        print(f"   评测模型: {', '.join(c.model_name for c in comparison_result.model_comparisons)}")
        print(f"   平均延迟范围: {min(c.avg_latency for c in comparison_result.model_comparisons):.1f}s - " +
              f"{max(c.avg_latency for c in comparison_result.model_comparisons):.1f}s")
        print(f"   成本范围: ¥{min(c.total_cost for c in comparison_result.model_comparisons):.4f} - " +
              f"¥{max(c.total_cost for c in comparison_result.model_comparisons):.4f}")
        
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
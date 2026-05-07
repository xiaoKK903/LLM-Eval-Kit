"""
Quality scoring module for LLM evaluation.

This module provides various quality scoring methods to evaluate
LLM response quality beyond simple latency and token counts.
"""

import re
from typing import Dict, Any, List, Optional


class QualityScorer:
    """Quality scoring class for LLM responses."""
    
    def __init__(self):
        self.scoring_methods = {
            "response_length": self.score_response_length,
            "structure_quality": self.score_structure_quality,
            "relevance": self.score_relevance,
            "completeness": self.score_completeness,
        }
    
    def score_response(self, prompt: str, response: str, expected_keywords: List[str] = None) -> Dict[str, float]:
        """Score a response using multiple quality metrics."""
        
        scores = {}
        
        # Apply all scoring methods
        for method_name, method_func in self.scoring_methods.items():
            scores[method_name] = method_func(prompt, response, expected_keywords)
        
        # Calculate overall score (weighted average)
        weights = {
            "response_length": 0.2,
            "structure_quality": 0.3,
            "relevance": 0.3,
            "completeness": 0.2,
        }
        
        overall_score = sum(scores[metric] * weight for metric, weight in weights.items())
        scores["overall"] = overall_score
        
        return scores
    
    def score_response_length(self, prompt: str, response: str, expected_keywords: List[str] = None) -> float:
        """Score based on response length appropriateness."""
        
        # Ideal response length range (characters)
        min_length = 100
        ideal_length = 300
        max_length = 1000
        
        length = len(response)
        
        if length < min_length:
            # Too short - linear penalty
            return max(0, length / min_length)
        elif length > max_length:
            # Too long - exponential penalty
            excess = length - max_length
            penalty = min(1.0, excess / 1000)  # Cap penalty at 1.0
            return max(0, 1.0 - penalty)
        else:
            # Within ideal range - bell curve scoring
            if length <= ideal_length:
                return length / ideal_length
            else:
                # Beyond ideal but within max
                return 1.0 - ((length - ideal_length) / (max_length - ideal_length)) * 0.5
    
    def score_structure_quality(self, prompt: str, response: str, expected_keywords: List[str] = None) -> float:
        """Score based on response structure and formatting."""
        
        score = 0.0
        
        # Check for structured formatting
        if "###" in response or "---" in response:
            score += 0.3  # Section headers
        
        if re.search(r"\\d+\\.\\s", response):  # Numbered lists
            score += 0.2
        
        if "* " in response or "- " in response:  # Bullet points
            score += 0.2
        
        if "**" in response or "__" in response:  # Emphasis
            score += 0.1
        
        # Check for logical structure
        paragraphs = response.split("\n\n")
        if len(paragraphs) >= 2:
            score += 0.2
        
        return min(1.0, score)
    
    def score_relevance(self, prompt: str, response: str, expected_keywords: List[str] = None) -> float:
        """Score based on relevance to the prompt."""
        
        if expected_keywords is None:
            # Extract keywords from prompt as fallback
            prompt_lower = prompt.lower()
            expected_keywords = []
            
            # Common business keywords
            business_terms = ["订单", "发货", "退货", "质量", "问题", "客服", "消费者", "权益"]
            for term in business_terms:
                if term in prompt_lower:
                    expected_keywords.append(term)
        
        if not expected_keywords:
            return 0.5  # Default score if no keywords
        
        response_lower = response.lower()
        matches = 0
        
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                matches += 1
        
        return min(1.0, matches / len(expected_keywords))
    
    def score_completeness(self, prompt: str, response: str, expected_keywords: List[str] = None) -> float:
        """Score based on completeness of the response."""
        
        # Check for common completeness indicators
        indicators = [
            ("总结", 0.2),  # Conclusion/summary
            ("建议", 0.3),  # Recommendations
            ("步骤", 0.2),  # Step-by-step
            ("原因", 0.2),  # Explanation of reasons
            ("法律", 0.1),  # Legal references
        ]
        
        score = 0.0
        response_lower = response.lower()
        
        for indicator, weight in indicators:
            if indicator in response_lower:
                score += weight
        
        # Bonus for comprehensive responses
        if len(response) > 500 and score > 0.5:
            score += 0.1
        
        return min(1.0, score)


def calculate_cost(tokens: int, model_name: str) -> float:
    """Calculate cost based on token usage and model pricing."""
    
    # Pricing per 1K tokens (in USD)
    pricing = {
        # Qwen pricing
        "qwen-turbo": {"input": 0.002, "output": 0.008},
        "qwen-plus": {"input": 0.01, "output": 0.02},
        "qwen-max": {"input": 0.04, "output": 0.08},
        
        # DeepSeek pricing
        "deepseek-chat": {"input": 0.0007, "output": 0.0028},
        "deepseek-coder": {"input": 0.0007, "output": 0.0028},
        "deepseek-v4-pro": {"input": 0.0014, "output": 0.0028},
        "deepseek-v4-flash": {"input": 0.00014, "output": 0.00056},
        
        # Default pricing (OpenAI-like)
        "default": {"input": 0.0015, "output": 0.002}
    }
    
    # Estimate input/output ratio (typically 1:4 for responses)
    input_tokens = tokens * 0.2  # 20% input tokens
    output_tokens = tokens * 0.8  # 80% output tokens
    
    # Find model pricing
    model_key = None
    for key in pricing:
        if key in model_name.lower():
            model_key = key
            break
    
    if model_key is None:
        model_key = "default"
    
    model_pricing = pricing[model_key]
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost
    
    return total_cost


if __name__ == "__main__":
    # Test the quality scorer
    scorer = QualityScorer()
    
    test_prompt = "我的订单为什么还没有发货？"
    test_response = """您的订单为什么还没有发货，可能有以下几种原因：

### 一、订单状态检查
1. **是否已付款？** - 请确认支付是否成功
2. **库存情况** - 商品可能暂时缺货

### 二、物流处理时间
- 通常24小时内处理
- 周末可能延迟

建议联系客服查询具体状态。"""
    
    scores = scorer.score_response(test_prompt, test_response, ["订单", "发货", "客服"])
    
    print("Quality Scores:")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Test cost calculation
    cost = calculate_cost(500, "qwen-turbo")
    print(f"\nCost for 500 tokens (qwen-turbo): ${cost:.4f}")
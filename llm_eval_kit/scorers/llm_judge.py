"""
LLM-based judge scoring for LLM responses.

This module provides an LLM-based scoring system that uses another LLM
to evaluate responses based on multiple criteria with position bias elimination.
"""

import json
import asyncio
import re
from typing import Dict, Any, Optional

from .base import BaseScorer, ScoreResult
from ..adapters.openai_compatible import OpenAICompatibleHttpxAdapter


class LLMJudgeScorer(BaseScorer):
    """LLM-based judge scorer using another LLM for evaluation."""
    
    def __init__(self, judge_config: Dict[str, Any]):
        """
        Initialize the LLM judge scorer.
        
        Args:
            judge_config: Configuration for the judge LLM
        """
        self.judge_config = judge_config
        self.judge_adapter = OpenAICompatibleHttpxAdapter(judge_config)
    
    def _create_judge_prompt(self, question: str, response: str, reference: str, 
                           order: str = "reference_first") -> str:
        """
        Create the judge prompt with position bias elimination.
        
        Args:
            question: The original question
            response: Model response to evaluate
            reference: Reference answer
            order: Order of presentation ("reference_first" or "response_first")
            
        Returns:
            Formatted judge prompt
        """
        if order == "reference_first":
            content_a = reference
            content_b = response
            order_desc = "参考答案在前，模型回答在后"
        else:
            content_a = response
            content_b = reference
            order_desc = "模型回答在前，参考答案在后"
        
        prompt = f"""你是一个专业的AI回答质量评估专家。请根据以下信息对模型回答进行评分。

问题：{question}

内容A：{content_a}

内容B：{content_b}

评估说明：
- 请从多个维度对内容B的质量进行评分（1-5分）
- 准确性：内容B是否准确反映了参考答案的关键信息
- 完整性：内容B是否完整回答了问题的所有要点
- 简洁性：内容B是否简洁明了，没有冗余信息
- 整体评分：基于以上维度的综合评分

评分规则：
- 5分：优秀，完全满足要求
- 4分：良好，基本满足要求
- 3分：一般，有改进空间
- 2分：较差，需要较大改进
- 1分：很差，不符合要求

请以JSON格式返回评分结果，格式如下：
{{
  "accuracy": 4,
  "completeness": 3,
  "conciseness": 5,
  "reasoning": "详细说明评分理由",
  "overall": 4.0
}}

注意：评分顺序为{order_desc}，请确保评分不受顺序影响。"""
        
        return prompt
    
    def _parse_judge_response(self, text: str) -> Dict[str, Any]:
        """
        Parse the judge LLM's response.
        
        Args:
            text: Raw response text from judge
            
        Returns:
            Parsed scoring result
            
        Raises:
            Exception: If parsing fails
        """
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if not json_match:
            raise ValueError("No JSON found in judge response")
        
        try:
            json_str = json_match.group(0)
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        
        # Validate required fields
        required_fields = ["accuracy", "completeness", "conciseness", "overall"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")
            
            # Validate score range
            score = result[field]
            if not (1 <= score <= 5):
                raise ValueError(f"{field} score {score} out of range (1-5)")
        
        # Ensure reasoning field exists
        if "reasoning" not in result:
            result["reasoning"] = "未提供详细理由"
        
        return result
    
    async def _get_judge_score(self, question: str, response: str, reference: str, 
                             order: str) -> Dict[str, Any]:
        """
        Get score from judge LLM for a specific order.
        
        Args:
            question: The original question
            response: Model response to evaluate
            reference: Reference answer
            order: Order of presentation
            
        Returns:
            Scoring result from judge
        """
        prompt = self._create_judge_prompt(question, response, reference, order)
        
        try:
            judge_response = await self.judge_adapter.generate(prompt)
            result = self._parse_judge_response(judge_response.text)
            return result
            
        except Exception as e:
            # Return default scores on failure
            return {
                "accuracy": 3,
                "completeness": 3,
                "conciseness": 3,
                "reasoning": f"评分失败: {str(e)}",
                "overall": 3.0
            }
    
    async def score_async(self, question: str, response: str, reference: str) -> ScoreResult:
        """
        Asynchronously score a model response using LLM judge.
        
        Args:
            question: The original question asked
            response: The model's response
            reference: The reference (correct) answer
            
        Returns:
            ScoreResult containing the score and reasoning
        """
        # Run two evaluations with different orders to eliminate position bias
        tasks = [
            self._get_judge_score(question, response, reference, "reference_first"),
            self._get_judge_score(question, response, reference, "response_first")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Judge evaluation {i+1} failed: {result}")
                # Use default scores for failed evaluations
                valid_results.append({
                    "accuracy": 3,
                    "completeness": 3,
                    "conciseness": 3,
                    "overall": 3.0,
                    "reasoning": f"评分失败: {str(result)}"
                })
            else:
                valid_results.append(result)
        
        # Calculate average scores
        if len(valid_results) == 0:
            # Fallback if all evaluations failed
            avg_scores = {
                "accuracy": 3,
                "completeness": 3,
                "conciseness": 3,
                "overall": 3.0,
                "reasoning": "所有评分尝试均失败"
            }
        else:
            avg_scores = {
                "accuracy": sum(r["accuracy"] for r in valid_results) / len(valid_results),
                "completeness": sum(r["completeness"] for r in valid_results) / len(valid_results),
                "conciseness": sum(r["conciseness"] for r in valid_results) / len(valid_results),
                "overall": sum(r["overall"] for r in valid_results) / len(valid_results),
                "reasoning": " | ".join(r["reasoning"] for r in valid_results)
            }
        
        # Convert 1-5 scale to 0-1 scale
        total_score = avg_scores["overall"] / 5.0
        
        # Prepare detailed breakdown
        details = {
            "accuracy_score": avg_scores["accuracy"] / 5.0,
            "completeness_score": avg_scores["completeness"] / 5.0,
            "conciseness_score": avg_scores["conciseness"] / 5.0,
            "raw_scores": avg_scores,
            "evaluations_count": len(valid_results)
        }
        
        return ScoreResult(
            total_score=total_score,
            details=details,
            reasoning=avg_scores["reasoning"]
        )
    
    def score(self, question: str, response: str, reference: str) -> ScoreResult:
        """
        Score a model response using LLM judge (synchronous wrapper).
        
        Args:
            question: The original question asked
            response: The model's response
            reference: The reference (correct) answer
            
        Returns:
            ScoreResult containing the score and reasoning
        """
        return asyncio.run(self.score_async(question, response, reference))
    
    async def close(self):
        """Close the judge adapter."""
        await self.judge_adapter.close()
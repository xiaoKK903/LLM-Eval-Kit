"""
Rule-based scoring for LLM responses.

This module provides a rule-based scoring system that evaluates responses
based on keyword matching and length appropriateness.
"""

import re
import jieba
from typing import Dict, Any

from .base import BaseScorer, ScoreResult


class RuleScorer(BaseScorer):
    """Rule-based scorer using keyword matching and length analysis."""
    
    def __init__(self, keyword_weight: float = 0.8, length_weight: float = 0.2):
        """
        Initialize the rule scorer.
        
        Args:
            keyword_weight: Weight for keyword matching score (default: 0.8)
            length_weight: Weight for length appropriateness score (default: 0.2)
        """
        self.keyword_weight = keyword_weight
        self.length_weight = length_weight
        
        # Validate weights
        if abs(keyword_weight + length_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
    
    def _extract_keywords(self, text: str) -> list:
        """
        Extract keywords from text using jieba segmentation.
        
        Args:
            text: Input text to extract keywords from
            
        Returns:
            List of keywords (excluding stop words and short words)
        """
        # Chinese stop words (simplified list)
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '他', '她', '它', '我们', '你们', '他们', '这个', '那个'
        }
        
        # Use jieba for Chinese word segmentation
        words = jieba.cut(text)
        
        # Filter out stop words and short words
        keywords = []
        for word in words:
            word = word.strip()
            if (len(word) >= 2 and 
                word not in stop_words and 
                not word.isdigit() and
                not all(c in '，。！？；：""''（）【】《》' for c in word)):
                keywords.append(word)
        
        return keywords
    
    def _calculate_keyword_score(self, response: str, reference: str) -> float:
        """
        Calculate keyword matching score.
        
        Args:
            response: Model response text
            reference: Reference answer text
            
        Returns:
            Keyword matching score (0-1)
        """
        # Extract keywords from reference
        ref_keywords = self._extract_keywords(reference)
        
        if not ref_keywords:
            # If no keywords found, return neutral score
            return 0.5
        
        # Extract keywords from response
        resp_keywords = self._extract_keywords(response)
        
        # Calculate matching score
        matched_count = 0
        for keyword in ref_keywords:
            if keyword in resp_keywords:
                matched_count += 1
        
        keyword_score = matched_count / len(ref_keywords)
        return min(1.0, keyword_score)
    
    def _calculate_length_score(self, response: str) -> float:
        """
        Calculate length appropriateness score.
        
        Args:
            response: Model response text
            
        Returns:
            Length score (0-1)
        """
        length = len(response.strip())
        
        # Score based on length appropriateness
        if length < 10:
            # Too short - linear penalty
            return max(0.0, length / 10)
        elif length > 500:
            # Too long - exponential penalty
            excess = length - 500
            penalty = min(1.0, excess / 1000)  # Cap penalty at 1.0
            return max(0.0, 1.0 - penalty)
        else:
            # Within ideal range
            if length <= 100:
                # 10-100 characters: linear scoring
                return 0.5 + (length - 10) / 180  # 0.5 to 1.0
            else:
                # 100-500 characters: slight penalty for longer responses
                return 1.0 - ((length - 100) / 400) * 0.2  # 1.0 to 0.8
    
    def score(self, question: str, response: str, reference: str) -> ScoreResult:
        """
        Score a model response using rule-based criteria.
        
        Args:
            question: The original question asked
            response: The model's response
            reference: The reference (correct) answer
            
        Returns:
            ScoreResult containing the score and reasoning
        """
        # Calculate individual scores
        keyword_score = self._calculate_keyword_score(response, reference)
        length_score = self._calculate_length_score(response)
        
        # Calculate weighted total score
        total_score = (keyword_score * self.keyword_weight + 
                      length_score * self.length_weight)
        
        # Prepare detailed breakdown
        details = {
            "keyword_score": keyword_score,
            "length_score": length_score,
            "keyword_weight": self.keyword_weight,
            "length_weight": self.length_weight,
            "response_length": len(response.strip())
        }
        
        # Generate reasoning
        reasoning_parts = []
        
        # Keyword reasoning
        if keyword_score >= 0.8:
            reasoning_parts.append("关键词匹配良好")
        elif keyword_score >= 0.5:
            reasoning_parts.append("关键词匹配一般")
        else:
            reasoning_parts.append("关键词匹配较差")
        
        # Length reasoning
        length = len(response.strip())
        if length < 10:
            reasoning_parts.append("回答过短")
        elif length > 500:
            reasoning_parts.append("回答过长")
        else:
            reasoning_parts.append("回答长度适中")
        
        reasoning = "; ".join(reasoning_parts)
        
        return ScoreResult(
            total_score=total_score,
            details=details,
            reasoning=reasoning
        )
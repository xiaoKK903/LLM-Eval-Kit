"""
Base adapter for LLM model interfaces.

This module defines the base interface that all model adapters should implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """Represents a response from an LLM model."""
    
    text: str
    """The generated text content."""
    
    latency: float
    """Response latency in seconds."""
    
    token_usage: Dict[str, int]
    """Token usage information (prompt_tokens, completion_tokens, total_tokens)."""
    
    model: str
    """The model name that generated the response."""


class BaseAdapter(ABC):
    """Base class for all LLM model adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Configuration dictionary for the model
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        pass
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt text
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse containing the generated text and metadata
        """
        pass
    
    @abstractmethod
    def get_cost(self, token_usage: Dict[str, int]) -> float:
        """
        Calculate the cost for the given token usage.
        
        Args:
            token_usage: Token usage dictionary
            
        Returns:
            Cost in USD
        """
        pass
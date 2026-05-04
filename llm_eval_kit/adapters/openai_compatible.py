"""
OpenAI-compatible model adapter.

This adapter supports any LLM provider that implements the OpenAI API interface,
including OpenAI, Anthropic (via proxy), and many open-source models.
"""

import time
from typing import Dict, Any, Optional
import httpx
from openai import OpenAI

from .base import BaseAdapter, ModelResponse


class OpenAICompatibleAdapter(BaseAdapter):
    """Adapter for OpenAI-compatible LLM APIs."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI-compatible adapter.
        
        Args:
            config: Configuration dictionary with:
                - base_url: API base URL (default: OpenAI official)
                - api_key: API key for authentication
                - model: Model name to use
                - timeout: Request timeout in seconds (default: 60)
        """
        super().__init__(config)
        
        # Initialize OpenAI client with minimal configuration
        # Use simpler approach to avoid compatibility issues
        api_key = self.config.get("api_key")
        base_url = self.config.get("base_url", "https://api.openai.com/v1")
        
        # Create client with minimal configuration
        if base_url != "https://api.openai.com/v1":
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        
        self.model_name = self.config.get("model", "gpt-3.5-turbo")
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        required_fields = ["api_key", "model"]
        
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")
        
        if not isinstance(self.config["api_key"], str):
            raise ValueError("api_key must be a string")
        
        if not isinstance(self.config["model"], str):
            raise ValueError("model must be a string")
        
        # Validate optional fields
        if "base_url" in self.config and not isinstance(self.config["base_url"], str):
            raise ValueError("base_url must be a string")
        
        if "timeout" in self.config and not isinstance(self.config["timeout"], (int, float)):
            raise ValueError("timeout must be a number")
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate a response for the given prompt using OpenAI-compatible API.
        
        Args:
            prompt: The input prompt text
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            ModelResponse containing the generated text and metadata
            
        Raises:
            Exception: If the API call fails
        """
        start_time = time.time()
        
        try:
            # Prepare the chat completion request
            messages = [{"role": "user", "content": prompt}]
            
            # Merge default parameters with user-provided ones
            completion_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
            }
            
            # Make the API call
            response = self.client.chat.completions.create(**completion_params)
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Extract token usage
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return ModelResponse(
                text=response_text,
                latency=latency,
                token_usage=token_usage,
                model=self.model_name
            )
            
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"OpenAI API call failed: {str(e)}")
    
    def get_cost(self, token_usage: Dict[str, int]) -> float:
        """
        Calculate the cost for the given token usage.
        
        Note: This is a simplified cost calculation. In practice, you would
        need to implement actual pricing based on the specific model.
        
        Args:
            token_usage: Token usage dictionary
            
        Returns:
            Cost in USD (estimated)
        """
        # Default pricing (GPT-3.5-turbo rates)
        # These are approximate and should be adjusted per model
        input_price_per_1k = 0.0015  # $0.0015 per 1K input tokens
        output_price_per_1k = 0.002   # $0.002 per 1K output tokens
        
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        input_cost = (prompt_tokens / 1000) * input_price_per_1k
        output_cost = (completion_tokens / 1000) * output_price_per_1k
        
        return input_cost + output_cost
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        base_url = self.config.get("base_url", "https://api.openai.com/v1")
        return f"OpenAICompatibleAdapter(model={self.model_name}, base_url={base_url})"
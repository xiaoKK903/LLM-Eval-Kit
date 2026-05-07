"""
OpenAI-compatible model adapter using httpx directly.

This adapter uses httpx to make direct API calls, avoiding compatibility issues
with the OpenAI client library.
"""

import time
import asyncio
import json
from typing import Dict, Any, Optional
import httpx

from .base import BaseAdapter, ModelResponse


class OpenAICompatibleHttpxAdapter(BaseAdapter):
    """Adapter for OpenAI-compatible LLM APIs using httpx directly."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI-compatible adapter.
        
        Args:
            config: Configuration dictionary with:
                - base_url: API base URL
                - api_key: API key for authentication
                - model: Model name to use
                - timeout: Request timeout in seconds (default: 30)
                - max_retries: Maximum number of retries (default: 3)
                - retry_delay: Base delay between retries in seconds (default: 1)
        """
        super().__init__(config)
        
        self.base_url = self.config.get("base_url", "https://api.openai.com/v1")
        self.api_key = self.config.get("api_key")
        self.model_name = self.config.get("model", "gpt-3.5-turbo")
        self.timeout = self.config.get("timeout", 30.0)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
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
        
        if "max_retries" in self.config and not isinstance(self.config["max_retries"], int):
            raise ValueError("max_retries must be an integer")
        
        if "retry_delay" in self.config and not isinstance(self.config["retry_delay"], (int, float)):
            raise ValueError("retry_delay must be a number")
    
    def _should_retry(self, status_code: int, error_message: str) -> bool:
        """
        Determine if a request should be retried based on the response.
        
        Args:
            status_code: HTTP status code
            error_message: Error message from the response
            
        Returns:
            True if the request should be retried, False otherwise
        """
        error_str = error_message.lower()
        
        # Retry on network/timeout errors
        if any(keyword in error_str for keyword in ["timeout", "timed out", "connection", "network"]):
            return True
        
        # Retry on rate limiting (429)
        if status_code == 429 or "rate limit" in error_str:
            return True
        
        # Retry on server errors (5xx)
        if 500 <= status_code < 600:
            return True
        
        # Do not retry on client errors
        if status_code in [401, 403] or "invalid api key" in error_str or "authentication" in error_str:
            return False
        
        # Do not retry on bad request errors
        if status_code == 400 or "bad request" in error_str or "invalid parameter" in error_str:
            return False
        
        # Default: retry on other errors
        return True
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate a response for the given prompt using httpx directly.
        
        Args:
            prompt: The input prompt text
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            ModelResponse containing the generated text and metadata
            
        Raises:
            Exception: If the API call fails after all retries
        """
        max_retries = self.max_retries
        retry_delay = self.retry_delay
        
        for attempt in range(max_retries + 1):  # +1 for the initial attempt
            start_time = time.time()
            
            try:
                # Prepare the request payload
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000),
                }
                
                # Make the API call
                response = await self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                )
                
                end_time = time.time()
                latency = end_time - start_time
                
                if response.status_code == 200:
                    # Parse successful response
                    data = response.json()
                    
                    # Extract response text
                    response_text = data["choices"][0]["message"]["content"]
                    
                    # Extract token usage
                    token_usage = {
                        "prompt_tokens": data["usage"]["prompt_tokens"],
                        "completion_tokens": data["usage"]["completion_tokens"],
                        "total_tokens": data["usage"]["total_tokens"]
                    }
                    
                    return ModelResponse(
                        text=response_text,
                        latency=latency,
                        token_usage=token_usage,
                        model=self.model_name
                    )
                else:
                    # Handle error response
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                    
                    # Check if we should retry
                    if attempt < max_retries and self._should_retry(response.status_code, error_message):
                        # Calculate exponential backoff delay
                        delay = retry_delay * (2 ** attempt)  # 1s, 2s, 4s, etc.
                        
                        print(f"Attempt {attempt + 1} failed: {error_message[:100]}... Retrying in {delay:.1f}s")
                        
                        # Wait before retrying
                        await asyncio.sleep(delay)
                        continue
                    else:
                        # Final attempt failed or should not retry
                        if attempt == max_retries:
                            error_msg = f"API call failed after {max_retries + 1} attempts: {error_message}"
                        else:
                            error_msg = f"API call failed (no retry): {error_message}"
                        
                        raise Exception(error_msg)
                        
            except httpx.TimeoutException as e:
                end_time = time.time()
                latency = end_time - start_time
                
                # Check if we should retry timeout errors
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} timed out: Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    raise Exception(f"API call timed out after {max_retries + 1} attempts")
                    
            except Exception as e:
                end_time = time.time()
                latency = end_time - start_time
                
                # Check if we should retry
                if attempt < max_retries and self._should_retry(0, str(e)):
                    delay = retry_delay * (2 ** attempt)
                    print(f"Attempt {attempt + 1} failed: {str(e)[:100]}... Retrying in {delay:.1f}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    if attempt == max_retries:
                        error_msg = f"API call failed after {max_retries + 1} attempts: {str(e)}"
                    else:
                        error_msg = f"API call failed (no retry): {str(e)}"
                    
                    raise Exception(error_msg)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def get_cost(self, token_usage: Dict[str, int]) -> float:
        """
        Calculate the cost for the given token usage.
        
        Args:
            token_usage: Token usage dictionary
            
        Returns:
            Cost in USD (estimated)
        """
        # Default pricing (GPT-3.5-turbo rates)
        input_price_per_1k = 0.0015  # $0.0015 per 1K input tokens
        output_price_per_1k = 0.002   # $0.002 per 1K output tokens
        
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        input_cost = (prompt_tokens / 1000) * input_price_per_1k
        output_cost = (completion_tokens / 1000) * output_price_per_1k
        
        return input_cost + output_cost
    
    def __str__(self) -> str:
        """String representation of the adapter."""
        return f"OpenAICompatibleHttpxAdapter(model={self.model_name}, base_url={self.base_url})"
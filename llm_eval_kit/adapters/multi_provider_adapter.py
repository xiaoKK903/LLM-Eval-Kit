"""
Multi-provider model adapter supporting various LLM APIs.

This adapter supports multiple LLM providers with their specific API formats
while maintaining a unified interface for evaluation.
"""

import time
import asyncio
import json
from typing import Dict, Any, Optional, List
import httpx

from .base import BaseAdapter, ModelResponse


class MultiProviderAdapter(BaseAdapter):
    """Adapter for multiple LLM providers with unified interface."""
    
    # Provider-specific API configurations
    PROVIDER_CONFIGS = {
        # OpenAI and compatible providers
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "endpoint": "/chat/completions",
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            "request_format": lambda prompt, model: {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "token_parser": lambda data: data.get("usage", {})
        },
        
        # Alibaba Qwen (DashScope)
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "endpoint": "/chat/completions",
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            "request_format": lambda prompt, model: {
                "model": model,
                "input": {
                    "messages": [{"role": "user", "content": prompt}]
                },
                "parameters": {
                    "temperature": 0.7,
                    "max_tokens": 1024
                }
            },
            "response_parser": lambda data: data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", ""),
            "token_parser": lambda data: data.get("usage", {})
        },
        
        # DeepSeek
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "endpoint": "/chat/completions",
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            "request_format": lambda prompt, model: {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False
            },
            "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "token_parser": lambda data: data.get("usage", {})
        },
        
        # Zhipu AI
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "endpoint": "/chat/completions",
            "headers": lambda api_key: {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            "request_format": lambda prompt, model: {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "response_parser": lambda data: data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "token_parser": lambda data: data.get("usage", {})
        },
        
        # Baidu ERNIE
        "ernie": {
            "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat",
            "endpoint": "/",
            "headers": lambda api_key: {
                "Content-Type": "application/json"
            },
            "request_format": lambda prompt, model: {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "response_parser": lambda data: data.get("result", ""),
            "token_parser": lambda data: {
                "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": data.get("usage", {}).get("total_tokens", 0)
            }
        },
        
        # iFlytek Spark
        "spark": {
            "base_url": "https://spark-api.xf-yun.com",
            "endpoint": "/v1.1/chat",
            "headers": lambda api_key: {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            },
            "request_format": lambda prompt, model: {
                "header": {
                    "app_id": api_key.split(".")[0] if api_key else "",
                    "uid": "user123"
                },
                "parameter": {
                    "chat": {
                        "domain": model,
                        "temperature": 0.7,
                        "max_tokens": 1024
                    }
                },
                "payload": {
                    "message": {
                        "text": [{"role": "user", "content": prompt}]
                    }
                }
            },
            "response_parser": lambda data: data.get("payload", {}).get("choices", {}).get("text", [{}])[0].get("content", ""),
            "token_parser": lambda data: {
                "prompt_tokens": data.get("usage", {}).get("text", [{}])[0].get("question_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("text", [{}])[0].get("completion_tokens", 0),
                "total_tokens": data.get("usage", {}).get("text", [{}])[0].get("total_tokens", 0)
            }
        }
    }
    
    # Model to provider mapping
    MODEL_PROVIDER_MAP = {
        # OpenAI models
        "gpt-4": "openai",
        "gpt-3.5-turbo": "openai",
        "gpt-4-turbo": "openai",
        
        # Qwen models
        "qwen-turbo": "qwen",
        "qwen-plus": "qwen",
        "qwen-max": "qwen",
        
        # DeepSeek models
        "deepseek-chat": "deepseek",
        "deepseek-coder": "deepseek",
        
        # Zhipu models
        "glm-4": "zhipu",
        "glm-3-turbo": "zhipu",
        
        # ERNIE models
        "ernie-bot": "ernie",
        "ernie-bot-turbo": "ernie",
        
        # Spark models
        "spark-v3": "spark",
        "spark-v2": "spark"
    }
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the multi-provider adapter.
        
        Args:
            config: Configuration dictionary with:
                - provider: Provider name (auto-detected if not provided)
                - base_url: API base URL (optional, uses default if not provided)
                - api_key: API key for authentication
                - model: Model name to use
                - timeout: Request timeout in seconds (default: 30)
                - max_retries: Maximum number of retries (default: 3)
                - retry_delay: Base delay between retries in seconds (default: 1)
        """
        super().__init__(config)
        
        self.model_name = self.config.get("model", "gpt-3.5-turbo")
        self.api_key = self.config.get("api_key")
        self.provider = self.config.get("provider")
        
        # Auto-detect provider if not specified
        if not self.provider:
            self.provider = self._detect_provider(self.model_name)
        
        # Get provider configuration
        provider_config = self.PROVIDER_CONFIGS.get(self.provider, self.PROVIDER_CONFIGS["openai"])
        
        # Set base URL (use config if provided, otherwise use provider default)
        self.base_url = self.config.get("base_url", provider_config["base_url"])
        self.endpoint = provider_config["endpoint"]
        self.timeout = self.config.get("timeout", 30.0)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
        # Store provider-specific functions
        self._get_headers = provider_config["headers"]
        self._format_request = provider_config["request_format"]
        self._parse_response = provider_config["response_parser"]
        self._parse_tokens = provider_config["token_parser"]
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers=self._get_headers(self.api_key)
        )
    
    def _detect_provider(self, model_name: str) -> str:
        """Detect the provider based on model name."""
        for model_pattern, provider in self.MODEL_PROVIDER_MAP.items():
            if model_pattern in model_name.lower():
                return provider
        
        # Default to OpenAI-compatible if no match
        return "openai"
    
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
        
        # Validate provider
        if "provider" in self.config and self.config["provider"] not in self.PROVIDER_CONFIGS:
            raise ValueError(f"Unsupported provider: {self.config['provider']}")
        
        # Validate optional fields
        if "base_url" in self.config and not isinstance(self.config["base_url"], str):
            raise ValueError("base_url must be a string")
        
        if "timeout" in self.config and not isinstance(self.config["timeout"], (int, float)):
            raise ValueError("timeout must be a number")
        
        if "max_retries" in self.config and not isinstance(self.config["max_retries"], int):
            raise ValueError("max_retries must be an integer")
        
        if "retry_delay" in self.config and not isinstance(self.config["retry_delay"], (int, float)):
            raise ValueError("retry_delay must be a number")
    
    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """
        Generate a response from the model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters for the request
            
        Returns:
            ModelResponse containing the generated text and metadata
        """
        start_time = time.time()
        
        # Format the request payload
        payload = self._format_request(prompt, self.model_name)
        
        # Merge additional parameters
        payload.update(kwargs)
        
        url = f"{self.base_url}{self.endpoint}"
        
        # Retry logic
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Parse response
                    text = self._parse_response(data)
                    token_usage = self._parse_tokens(data)
                    
                    # Calculate latency
                    latency = time.time() - start_time
                    
                    return ModelResponse(
                        text=text,
                        latency=latency,
                        token_usage=token_usage,
                        model=self.model_name
                    )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    
                    # Check if we should retry
                    if attempt < self.max_retries and self._should_retry(response.status_code, error_msg):
                        delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        await asyncio.sleep(delay)
                        continue
                    
                    raise Exception(error_msg)
                    
            except Exception as e:
                if attempt < self.max_retries and self._should_retry(0, str(e)):
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                
                raise Exception(f"Failed to generate response after {attempt + 1} attempts: {e}")
        
        raise Exception(f"Failed to generate response after {self.max_retries + 1} attempts")
    
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
        
        # Don't retry on client errors (4xx) except 429
        if 400 <= status_code < 500 and status_code != 429:
            return False
        
        return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    def get_cost(self, token_usage: Dict[str, int]) -> float:
        """
        Calculate the cost for the given token usage.
        
        Args:
            token_usage: Token usage dictionary
            
        Returns:
            Cost in USD (approximate conversion from Chinese Yuan)
        """
        # Extract token counts
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        # Use the same pricing logic as the comparator
        from ..reporter.comparator import ModelComparator
        comparator = ModelComparator()
        
        # Calculate cost in Chinese Yuan
        cost_cny = comparator._calculate_cost(self.model_name, prompt_tokens, completion_tokens)
        
        # Convert to USD (approximate conversion rate: 1 USD ≈ 7.2 CNY)
        cost_usd = cost_cny / 7.2
        
        return cost_usd
    
    @property
    def model(self) -> str:
        """Get the model name."""
        return self.model_name
    
    @classmethod
    def get_supported_models(cls) -> Dict[str, List[str]]:
        """Get a dictionary of supported models by provider."""
        supported_models = {}
        
        for model, provider in cls.MODEL_PROVIDER_MAP.items():
            if provider not in supported_models:
                supported_models[provider] = []
            supported_models[provider].append(model)
        
        return supported_models
    
    @classmethod
    def get_provider_config(cls, provider: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific provider."""
        return cls.PROVIDER_CONFIGS.get(provider)
"""
OpenAI-compatible model adapter with robust concurrency and retry.

Unified adapter for any OpenAI-compatible API (OpenAI, Qwen, DeepSeek, Ollama, etc.)
Features:
- Dual-layer Semaphore concurrency control
- Exponential backoff with jitter
- Intelligent retry (timeout, 429, 5xx)
- Token usage tracking
"""

import asyncio
import time
import random
from typing import Dict, Any, Optional

import httpx

from .base import BaseAdapter, ModelResponse


RETRYABLE_STATUSES = {429, 502, 503, 504}
NON_RETRYABLE_KEYWORDS = ["401", "403", "400", "auth", "invalid", "not found",
                          "insufficient_quota", "rate limit exceeded"]


class OpenAICompatibleAdapter(BaseAdapter):
    """Adapter for any OpenAI-compatible API endpoint."""

    def __init__(self, config: Dict[str, Any]):
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.api_key = config.get("api_key", "")
        self.model_name = config.get("model", "gpt-3.5-turbo")
        self.timeout = config.get("timeout", 60.0)
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1.0)
        self.max_concurrency = config.get("max_concurrency", 10)
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)

        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def _validate_config(self):
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.model_name:
            raise ValueError("model is required")

    def _is_retryable(self, status_code: int, error_message: str) -> bool:
        if status_code in RETRYABLE_STATUSES:
            return True
        if status_code >= 500:
            return True
        if "timeout" in error_message.lower():
            return True
        err_lower = error_message.lower()
        for keyword in NON_RETRYABLE_KEYWORDS:
            if keyword in err_lower:
                return False
        return status_code == 0

    async def generate(self, prompt: str, **kwargs) -> ModelResponse:
        self._validate_config()

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        max_retries = kwargs.get("max_retries", self.max_retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        max_total_time = kwargs.get("max_total_time", 120.0)

        start_time = time.time()

        async with self._semaphore:
            for attempt in range(max_retries + 1):
                try:
                    response = await self.client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                    )
                    elapsed = time.time() - start_time

                    if response.status_code == 200:
                        data = response.json()
                        choice = data["choices"][0]
                        response_text = choice["message"]["content"].strip()

                        usage = data.get("usage", {})
                        token_usage = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        }

                        return ModelResponse(
                            text=response_text,
                            latency=elapsed,
                            token_usage=token_usage,
                            model=self.model_name,
                        )

                    error_msg = f"HTTP {response.status_code}: {response.text[:200]}"

                    if self._is_retryable(response.status_code, response.text):
                        if attempt < max_retries and elapsed < max_total_time:
                            delay = self._backoff_delay(attempt, retry_delay)
                            print(f"  [{self.model_name}] 重试 {attempt+1}/{max_retries} ({response.status_code}, 等待{delay:.1f}s)")
                            await asyncio.sleep(delay)
                            continue
                    raise Exception(error_msg)

                except httpx.TimeoutException:
                    elapsed = time.time() - start_time
                    if attempt < max_retries and elapsed < max_total_time:
                        delay = self._backoff_delay(attempt, retry_delay)
                        print(f"  [{self.model_name}] 重试 {attempt+1}/{max_retries} (超时, 等待{delay:.1f}s)")
                        await asyncio.sleep(delay)
                        continue
                    raise Exception(f"Timeout after {max_retries + 1} attempts ({elapsed:.1f}s)")

                except Exception as e:
                    elapsed = time.time() - start_time
                    if attempt < max_retries and self._is_retryable(0, str(e)) and elapsed < max_total_time:
                        delay = self._backoff_delay(attempt, retry_delay)
                        print(f"  [{self.model_name}] 重试 {attempt+1}/{max_retries}: {str(e)[:80]} (等待{delay:.1f}s)")
                        await asyncio.sleep(delay)
                        continue
                    raise Exception(f"API failed after {attempt + 1} attempt(s): {str(e)[:200]}")

    def _backoff_delay(self, attempt: int, base_delay: float) -> float:
        delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0, delay * 0.3)
        return delay + jitter

    async def close(self):
        await self.client.aclose()

    def get_cost(self, token_usage: Dict[str, int]) -> float:
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        from ..utils.cost_calc import calculate_cost_usd
        return calculate_cost_usd(self.model_name, prompt_tokens, completion_tokens)

    def __str__(self):
        return f"OpenAICompatibleAdapter(model={self.model_name})"

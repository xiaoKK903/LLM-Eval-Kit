"""
多模型对比评测示例 — 同时对比多个模型的评分、成本、延迟
"""

import os
import asyncio
from llm_eval_kit import Evaluator


async def main():
    models = [
        {
            "model": "deepseek-chat",
            "api_key": os.getenv("DEEPSEEK_API_KEY", "your-key-here"),
            "base_url": "https://api.deepseek.com/v1",
        },
        {
            "model": "qwen-turbo",
            "api_key": os.getenv("QWEN_API_KEY", "your-key-here"),
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        },
        {
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY", "your-key-here"),
            "base_url": "https://api.openai.com/v1",
        },
    ]

    evaluator = Evaluator()
    result = await evaluator.run(
        models=models,
        data_path="examples/sample_data.jsonl",
        concurrency=3,
        output_path="eval_report.json",
    )

    print(f"\n最佳综合: {result.best_overall_model} ({result.best_overall_score:.4f})")
    print(f"最快响应: {result.fastest_model} ({result.fastest_latency:.2f}s)")
    print(f"最低成本: {result.cheapest_model}")
    print(f"最高性价比: {result.best_value_model}")


if __name__ == "__main__":
    asyncio.run(main())

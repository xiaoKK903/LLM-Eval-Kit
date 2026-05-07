"""
LLM-Eval-Kit 三行代码评测示例

用法:
    python examples/basic_usage.py

前提:
    1. 设置环境变量 DEEPSEEK_API_KEY
    2. 可选: 设置 QWEN_API_KEY 做多模型对比
"""

import os
import asyncio
from llm_eval_kit import Evaluator


async def main():
    # 配置模型（只配一个也能跑）
    models = [
        {
            "model": "deepseek-chat",
            "api_key": os.getenv("DEEPSEEK_API_KEY", "your-key-here"),
            "base_url": "https://api.deepseek.com/v1",
        },
    ]

    # 三行核心代码
    evaluator = Evaluator()
    result = await evaluator.run(models=models, data_path="examples/sample_data.jsonl")

    print("评测完成！")
    print(f"最佳模型: {result.best_overall_model}")


if __name__ == "__main__":
    asyncio.run(main())

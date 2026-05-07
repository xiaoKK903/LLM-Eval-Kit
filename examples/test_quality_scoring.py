"""
Test script for quality scoring and cost calculation features.
"""

import asyncio
import os

from llm_eval_kit import RuleScorer, OpenAICompatibleAdapter
from llm_eval_kit.utils.cost_calc import calculate_cost_cny, calculate_cost_usd, format_cost


async def test_quality_scoring():
    """Test the new quality scoring and cost calculation features."""

    print("Testing Quality Scoring & Cost Calculation")
    print("=" * 60)

    scorer = RuleScorer()

    test_cases = [
        {
            "question": "我的订单为什么还没有发货？",
            "response": """您的订单为什么还没有发货，可能有以下几种原因：

### 一、订单状态检查
1. **是否已付款？** - 请确认支付是否成功
2. **库存情况** - 商品可能暂时缺货
3. **物流信息** - 部分物流更新有延迟

建议您先检查订单详情页，或联系在线客服获取最新进度。""",
            "reference": "检查订单状态，确认支付和库存情况，联系客服获取最新进度",
        },
        {
            "question": "退货流程是什么？",
            "response": "不知道。",
            "reference": "联系客服→提交退货申请→寄回商品→商家验收→退款",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        result = await scorer.score(case["question"], case["response"], case["reference"])
        print(f"\n用例 {i}:")
        print(f"  综合分: {result.total_score:.4f}")
        print(f"  详情: {result.details}")
        print(f"  说明: {result.reasoning}")

    print("\n" + "=" * 60)
    print("Cost Calculation Demo:")
    print("=" * 60)

    test_costs = [
        ("qwen-turbo", 150, 350),
        ("deepseek-chat", 200, 500),
        ("gpt-4", 300, 700),
    ]

    print(f"{'模型':<20} {'输入':>6} {'输出':>6} {'成本(¥)':>12} {'成本($)':>12}")
    print("-" * 56)
    for model, inp, out in test_costs:
        cny = calculate_cost_cny(model, inp, out)
        usd = calculate_cost_usd(model, inp, out)
        print(f"{model:<20} {inp:>6} {out:>6} {format_cost(cny):>12} {format_cost(usd):>12}")
    print()


if __name__ == "__main__":
    asyncio.run(test_quality_scoring())

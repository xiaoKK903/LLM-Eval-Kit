"""
规则评分器独立使用示例 — 无需 API Key，纯本地运行
"""

import asyncio
from llm_eval_kit import RuleScorer


async def main():
    scorer = RuleScorer()

    test_cases = [
        {
            "question": "订单还没发货怎么办？",
            "response": "您好，建议您先检查订单状态是否已支付。如已支付超过48小时未发货，请联系客服并提供订单号，我们会尽快为您处理发货事宜。",
            "reference": "检查订单状态，已支付超48小时未发货请联系客服",
        },
        {
            "question": "退货流程是什么？",
            "response": "好的。",
            "reference": "联系客服→提交退货申请→寄回商品→商家验收→退款",
        },
    ]

    for i, case in enumerate(test_cases, 1):
        result = await scorer.score(case["question"], case["response"], case["reference"])
        print(f"\n用例 {i}:")
        print(f"  综合分: {result.total_score:.4f}")
        print(f"  详情: {result.details}")
        print(f"  说明: {result.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())

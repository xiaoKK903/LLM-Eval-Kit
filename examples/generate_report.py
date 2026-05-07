"""
HTML Report Generation Demo for LLM-Eval-Kit

This script demonstrates how to generate a standalone HTML evaluation report
with comparison tables, conclusion cards, and per-sample details.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval_kit.reporter.models import EvaluationResult
from llm_eval_kit.reporter.html_reporter import HtmlReporter


def create_mock_data():
    """Create mock evaluation results for demonstration."""
    mock_results = []

    qwen_responses = [
        "退款通常需要3-5个工作日处理完成。",
        "您可以在设置页面的安全选项中修改密码。",
        "客服电话是400-123-4567，工作时间9:00-18:00。",
    ]

    deepseek_responses = [
        "退款处理时间一般为3到5个工作日，具体时间可能因银行而异。",
        "修改密码的方法：进入设置页面，找到安全选项，点击修改密码进行设置。",
        "我们的客服电话是400-123-4567，如有问题欢迎随时联系。",
    ]

    samples = [
        {"id": "001", "question": "退款要多久？", "reference": "3-5个工作日"},
        {"id": "002", "question": "如何修改密码？", "reference": "在设置页面找到安全选项，点击修改密码"},
        {"id": "003", "question": "客服电话是多少？", "reference": "400-123-4567"},
    ]

    for i, sample in enumerate(samples):
        qwen_result = EvaluationResult(
            sample_id=sample["id"],
            question=sample["question"],
            response=qwen_responses[i],
            latency=6.2 + i * 0.5,
            token_usage={
                "prompt_tokens": 120 + i * 20,
                "completion_tokens": 180 + i * 30,
                "total_tokens": 300 + i * 50,
            },
            model="qwen-turbo",
            scoring_result={
                "total_score": 0.85 + i * 0.05,
                "reasoning": "关键词匹配良好，回答长度适中",
                "details": {
                    "keyword_score": 0.9 + i * 0.02,
                    "length_score": 0.8 + i * 0.03,
                },
            },
        )
        mock_results.append(qwen_result)

        deepseek_result = EvaluationResult(
            sample_id=sample["id"],
            question=sample["question"],
            response=deepseek_responses[i],
            latency=10.8 + i * 0.8,
            token_usage={
                "prompt_tokens": 120 + i * 20,
                "completion_tokens": 250 + i * 40,
                "total_tokens": 370 + i * 60,
            },
            model="deepseek-chat",
            scoring_result={
                "total_score": 0.88 + i * 0.04,
                "reasoning": "回答详细准确，信息完整",
                "details": {
                    "keyword_score": 0.85 + i * 0.03,
                    "length_score": 0.9 + i * 0.02,
                },
            },
        )
        mock_results.append(deepseek_result)

    return mock_results


def main():
    print("=" * 60)
    print("  LLM-Eval-Kit: HTML 报告生成演示")
    print("=" * 60)

    print("\n📦 创建模拟评测数据...")
    results = create_mock_data()
    print(f"   共 {len(results)} 条评测结果（2个模型 × 3条样本）")

    print("\n🔧 初始化 HTML 报告生成器...")
    reporter = HtmlReporter()

    print("\n📄 生成 HTML 报告...")
    output_path = reporter.generate_report(
        results=results,
        data_name="客服问答测试集",
        output_dir="reports",
    )

    print(f"\n✅ 报告已生成: {output_path}")
    print(f"\n📋 报告包含以下内容：")
    print(f"   • 报告头部（评测概览）")
    print(f"   • 模型对比总览表格")
    print(f"   • 四张结论卡片（综合最优/速度最快/成本最低/性价比王）")
    print(f"   • 每个样本的详细结果（问题、参考答案、各模型回答）")

    print(f"\n🌐 请在浏览器中打开该 HTML 文件查看报告。")

    abs_path = os.path.abspath(output_path)
    print(f"\n   文件路径: file:///{abs_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()

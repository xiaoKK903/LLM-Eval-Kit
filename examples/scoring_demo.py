"""
Scoring Demo for LLM-Eval-Kit

This script demonstrates the scoring functionality of the evaluation toolkit,
comparing rule-based scoring with LLM-based judging.
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval_kit.scorers import RuleScorer, LLMJudgeScorer
from llm_eval_kit.scorers.base import ScoreResult


async def demo_rule_scorer():
    """Demonstrate rule-based scoring."""
    print("🔍 规则评分器演示")
    print("=" * 60)
    
    # Initialize rule scorer
    scorer = RuleScorer(keyword_weight=0.8, length_weight=0.2)
    
    # Test data
    test_samples = [
        {
            "id": "001",
            "question": "退款要多久？",
            "reference": "3-5个工作日",
            "response": "您好，退款将在3到5个工作日内处理完成。"
        },
        {
            "id": "002", 
            "question": "如何修改密码？",
            "reference": "在设置页面找到安全选项，点击修改密码",
            "response": "请到个人中心的设置菜单里修改您的登录密码。"
        },
        {
            "id": "003",
            "question": "客服电话是多少？",
            "reference": "400-123-4567",
            "response": "很抱歉，我无法提供具体的客服电话信息。"
        }
    ]
    
    for sample in test_samples:
        print(f"\n📋 样本 {sample['id']}:")
        print(f"   问题: {sample['question']}")
        print(f"   参考答案: {sample['reference']}")
        print(f"   模型回答: {sample['response']}")
        
        # Score the response
        result = scorer.score(
            sample["question"],
            sample["response"], 
            sample["reference"]
        )
        
        print(f"\n   📊 评分结果:")
        print(f"     总分: {result.total_score:.3f}")
        print(f"     关键词得分: {result.details['keyword_score']:.3f}")
        print(f"     长度得分: {result.details['length_score']:.3f}")
        print(f"     回答长度: {result.details['response_length']} 字符")
        print(f"     理由: {result.reasoning}")
    
    print("\n" + "=" * 60)


async def demo_llm_judge():
    """Demonstrate LLM-based judging."""
    print("\n🤖 LLM Judge评分器演示")
    print("=" * 60)
    
    # Check if API keys are available
    qwen_api_key = os.getenv("QWEN_API_KEY")
    if not qwen_api_key:
        print("⚠️  未找到QWEN_API_KEY，跳过LLM Judge演示")
        print("   请在.env文件中设置QWEN_API_KEY环境变量")
        return
    
    # Initialize LLM judge scorer
    judge_config = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": qwen_api_key,
        "model": "qwen-turbo",
        "timeout": 30,
        "max_retries": 3
    }
    
    try:
        scorer = LLMJudgeScorer(judge_config)
        
        # Test data
        test_samples = [
            {
                "id": "001",
                "question": "退款要多久？",
                "reference": "3-5个工作日",
                "response": "您好，退款将在3到5个工作日内处理完成。"
            },
            {
                "id": "002",
                "question": "如何修改密码？", 
                "reference": "在设置页面找到安全选项，点击修改密码",
                "response": "请到个人中心的设置菜单里修改您的登录密码。"
            },
            {
                "id": "003",
                "question": "客服电话是多少？",
                "reference": "400-123-4567", 
                "response": "很抱歉，我无法提供具体的客服电话信息。"
            }
        ]
        
        for sample in test_samples:
            print(f"\n📋 样本 {sample['id']}:")
            print(f"   问题: {sample['question']}")
            print(f"   参考答案: {sample['reference']}")
            print(f"   模型回答: {sample['response']}")
            
            # Score the response using LLM judge
            result = await scorer.score_async(
                sample["question"],
                sample["response"],
                sample["reference"]
            )
            
            print(f"\n   📊 LLM Judge评分结果:")
            print(f"     总分: {result.total_score:.3f}")
            print(f"     准确性: {result.details['raw_scores']['accuracy']}/5")
            print(f"     完整性: {result.details['raw_scores']['completeness']}/5")
            print(f"     简洁性: {result.details['raw_scores']['conciseness']}/5")
            print(f"     整体评分: {result.details['raw_scores']['overall']}/5")
            print(f"     理由: {result.reasoning}")
        
        # Close the scorer
        await scorer.close()
        
    except Exception as e:
        print(f"❌ LLM Judge初始化失败: {e}")
        print("   请检查API密钥和网络连接")
    
    print("\n" + "=" * 60)


async def compare_scorers():
    """Compare both scoring methods on the same sample."""
    print("\n🔬 评分器对比演示")
    print("=" * 60)
    
    # Test sample
    sample = {
        "question": "退款要多久？",
        "reference": "3-5个工作日",
        "response": "您好，退款处理通常需要3到5个工作日，具体时间可能因银行处理速度而有所不同。"
    }
    
    print(f"问题: {sample['question']}")
    print(f"参考答案: {sample['reference']}")
    print(f"模型回答: {sample['response']}")
    
    # Rule-based scoring
    rule_scorer = RuleScorer()
    rule_result = rule_scorer.score(
        sample["question"],
        sample["response"],
        sample["reference"]
    )
    
    print(f"\n📋 规则评分器:")
    print(f"   总分: {rule_result.total_score:.3f}")
    print(f"   关键词得分: {rule_result.details['keyword_score']:.3f}")
    print(f"   长度得分: {rule_result.details['length_score']:.3f}")
    print(f"   理由: {rule_result.reasoning}")
    
    # LLM-based judging (if available)
    qwen_api_key = os.getenv("QWEN_API_KEY")
    if qwen_api_key:
        try:
            judge_config = {
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_key": qwen_api_key,
                "model": "qwen-turbo",
                "timeout": 30,
                "max_retries": 3
            }
            
            llm_scorer = LLMJudgeScorer(judge_config)
            llm_result = await llm_scorer.score_async(
                sample["question"],
                sample["response"],
                sample["reference"]
            )
            
            print(f"\n📋 LLM Judge评分器:")
            print(f"   总分: {llm_result.total_score:.3f}")
            print(f"   准确性: {llm_result.details['raw_scores']['accuracy']}/5")
            print(f"   完整性: {llm_result.details['raw_scores']['completeness']}/5")
            print(f"   简洁性: {llm_result.details['raw_scores']['conciseness']}/5")
            print(f"   整体评分: {llm_result.details['raw_scores']['overall']}/5")
            print(f"   理由: {llm_result.reasoning}")
            
            await llm_scorer.close()
            
        except Exception as e:
            print(f"\n📋 LLM Judge评分器: 不可用 ({e})")
    else:
        print(f"\n📋 LLM Judge评分器: 需要API密钥")
    
    print("\n" + "=" * 60)


async def main():
    """Main demonstration function."""
    print("🎯 LLM-Eval-Kit 评分引擎演示")
    print("=" * 60)
    print("演示两种评分方法：")
    print("1. 规则评分器 - 基于关键词匹配和长度分析")
    print("2. LLM Judge评分器 - 基于另一个LLM的多维度评估")
    print("=" * 60)
    
    # Run demonstrations
    await demo_rule_scorer()
    await demo_llm_judge()
    await compare_scorers()
    
    print("\n✅ 评分引擎演示完成!")
    print("\n💡 使用建议:")
    print("   • 规则评分器: 快速、低成本，适合批量初步筛选")
    print("   • LLM Judge评分器: 准确、多维度，适合关键场景深度评估")
    print("   • 组合使用: 先用规则评分器筛选，再用LLM Judge深度评估")


if __name__ == "__main__":
    asyncio.run(main())
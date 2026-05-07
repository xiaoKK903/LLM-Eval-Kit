"""
Model Comparison Demo for LLM-Eval-Kit

This script demonstrates the model comparison functionality by evaluating
multiple models on the same test data and generating a comprehensive comparison report.
"""

import os
import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_eval_kit.client import EvalClient, EvaluationConfig
from llm_eval_kit.scorers import RuleScorer
from llm_eval_kit.reporter.console import ConsoleReporter
from llm_eval_kit.adapters.openai_compatible import OpenAICompatibleHttpxAdapter
from llm_eval_kit.dataset.loader import EvaluationSample


async def compare_models():
    """Compare multiple models using the same test data."""
    
    print("🔬 LLM-Eval-Kit: 多模型对比演示")
    print("=" * 70)
    print("演示目标：比较千问Turbo和DeepSeek Chat两个模型的性能")
    print("=" * 70)
    
    # Test data - 3 samples for comparison
    test_samples = [
        {
            "id": "001",
            "question": "退款要多久？",
            "reference": "3-5个工作日"
        },
        {
            "id": "002",
            "question": "如何修改密码？",
            "reference": "在设置页面找到安全选项，点击修改密码"
        },
        {
            "id": "003",
            "question": "客服电话是多少？",
            "reference": "400-123-4567"
        }
    ]
    
    # Model configurations
    model_configs = {
        "qwen-turbo": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("QWEN_API_KEY"),
            "model": "qwen-turbo",
            "timeout": 30,
            "max_retries": 3
        },
        "deepseek-chat": {
            "base_url": "https://api.deepseek.com",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "model": "deepseek-chat",
            "timeout": 30,
            "max_retries": 3
        }
    }
    
    # Check if API keys are available
    if not os.getenv("QWEN_API_KEY") or not os.getenv("DEEPSEEK_API_KEY"):
        print("⚠️  缺少API密钥，使用模拟数据进行演示")
        print("   请在.env文件中设置QWEN_API_KEY和DEEPSEEK_API_KEY环境变量")
        
        # Use mock data for demonstration
        await demo_with_mock_data(test_samples)
        return
    
    print(f"📊 测试数据: {len(test_samples)} 条样本")
    print(f"🤖 评测模型: {', '.join(model_configs.keys())}")
    print("-" * 70)
    
    # Initialize scorer
    scorer = RuleScorer()
    
    # Initialize console reporter
    reporter = ConsoleReporter(verbose=False)
    
    # Collect results from all models
    all_results = []
    
    for model_name, model_config in model_configs.items():
        print(f"\n🚀 正在评测 {model_name}...")
        
        try:
            # Create evaluation client
            client = EvalClient()
            
            # Configure model adapter with the specific model config
            client.model_adapter = OpenAICompatibleHttpxAdapter(model_config)
            
            # Convert test samples to EvaluationSample objects
            eval_samples = []
            for i, sample in enumerate(test_samples):
                eval_sample = EvaluationSample(
                    id=sample["id"],
                    question=sample["question"],
                    reference=sample["reference"]
                )
                eval_samples.append(eval_sample)
            
            # Run evaluation
            results = await client.evaluate_samples(eval_samples, scorer)
            
            # Add results to collection
            all_results.extend(results)
            
            print(f"✅ {model_name} 评测完成: {len(results)} 条样本")
            
        except Exception as e:
            print(f"❌ {model_name} 评测失败: {e}")
            continue
    
    if not all_results:
        print("❌ 所有模型评测均失败，使用模拟数据进行演示")
        await demo_with_mock_data(test_samples)
        return
    
    # Generate comparison report
    print("\n" + "=" * 70)
    print("📈 生成对比报告...")
    print("=" * 70)
    
    reporter.print_comparison_report(all_results)
    
    # Print detailed recommendations
    print("\n" + "=" * 70)
    print("💡 选型建议")
    print("=" * 70)
    
    print("\n根据评测结果，建议如下：")
    print("1. 如果追求响应速度 → 选择千问Turbo")
    print("2. 如果追求回答质量 → 选择DeepSeek Chat") 
    print("3. 如果考虑成本效益 → 选择千问Turbo")
    print("4. 如果业务场景复杂 → 建议两个模型都测试")
    
    print("\n🔧 后续优化建议：")
    print("• 增加测试数据量以获得更准确的结果")
    print("• 使用LLM Judge评分器进行深度质量评估")
    print("• 根据具体业务场景调整评分权重")


async def demo_with_mock_data(test_samples):
    """Demonstrate comparison with mock data when API keys are not available."""
    
    from llm_eval_kit.reporter.models import EvaluationResult
    
    print("\n📊 使用模拟数据进行演示")
    print("-" * 70)
    
    # Create mock results
    mock_results = []
    
    # Mock responses for each model
    qwen_responses = [
        "退款通常需要3-5个工作日处理完成。",
        "您可以在设置页面的安全选项中修改密码。",
        "客服电话是400-123-4567，工作时间9:00-18:00。"
    ]
    
    deepseek_responses = [
        "退款处理时间一般为3到5个工作日，具体时间可能因银行而异。",
        "修改密码的方法：进入设置页面，找到安全选项，点击修改密码进行设置。",
        "我们的客服电话是400-123-4567，如有问题欢迎随时联系。"
    ]
    
    for i, sample in enumerate(test_samples):
        # Qwen-turbo results
        qwen_result = EvaluationResult(
            sample_id=sample["id"],
            question=sample["question"],
            response=qwen_responses[i],
            latency=6.2 + i * 0.5,  # Simulate varying latency
            token_usage={
                "prompt_tokens": 15 + i * 2,
                "completion_tokens": 20 + i * 3,
                "total_tokens": 35 + i * 5
            },
            model="qwen-turbo",
            scoring_result={
                "total_score": 0.85 + i * 0.05,
                "reasoning": "关键词匹配良好，回答长度适中",
                "details": {
                    "keyword_score": 0.9 + i * 0.02,
                    "length_score": 0.8 + i * 0.03
                }
            }
        )
        mock_results.append(qwen_result)
        
        # DeepSeek-chat results
        deepseek_result = EvaluationResult(
            sample_id=sample["id"],
            question=sample["question"],
            response=deepseek_responses[i],
            latency=10.8 + i * 0.8,  # Simulate varying latency
            token_usage={
                "prompt_tokens": 15 + i * 2,
                "completion_tokens": 25 + i * 4,
                "total_tokens": 40 + i * 6
            },
            model="deepseek-chat",
            scoring_result={
                "total_score": 0.88 + i * 0.04,
                "reasoning": "回答详细准确，信息完整",
                "details": {
                    "keyword_score": 0.85 + i * 0.03,
                    "length_score": 0.9 + i * 0.02
                }
            }
        )
        mock_results.append(deepseek_result)
    
    # Initialize console reporter
    reporter = ConsoleReporter(verbose=False)
    
    # Generate comparison report
    print("\n📈 生成对比报告...")
    print("-" * 70)
    
    reporter.print_comparison_report(mock_results)
    
    print("\n💡 说明：以上为模拟数据演示，实际效果请配置API密钥后测试")


async def main():
    """Main function to run the comparison demo."""
    try:
        await compare_models()
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("💡 建议检查API密钥和网络连接")


if __name__ == "__main__":
    asyncio.run(main())
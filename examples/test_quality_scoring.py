"""
Test script for quality scoring and cost calculation features.
"""

import asyncio
import os
from dotenv import load_dotenv

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation

# Load environment variables
load_dotenv()


async def test_quality_scoring():
    """Test the new quality scoring and cost calculation features."""
    
    print("🚀 Testing Quality Scoring & Cost Calculation")
    print("=" * 60)
    
    # Test with a single model to see detailed output
    model_config = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("QWEN_API_KEY"),
        "model": "qwen-turbo",
        "timeout": 30
    }
    
    print("Testing Qwen-Turbo with verbose mode to see quality scores...")
    
    try:
        # Run evaluation with verbose mode to see detailed output
        results = await run_evaluation(
            data_path="examples/sample_data.jsonl",
            model_config=model_config,
            verbose=True,  # Enable verbose mode to see quality scores
            max_samples=1   # Test with 1 sample for quick results
        )
        
        if results:
            result = results[0]
            print("\n✅ Evaluation completed successfully!")
            print(f"\n📊 Detailed Results:")
            print(f"   Sample ID: {result.sample_id}")
            print(f"   Latency: {result.latency:.2f}s")
            print(f"   Tokens: {result.token_usage.get('total_tokens', 0)}")
            
            # Check if quality scores are available
            if hasattr(result, 'quality_scores') and result.quality_scores:
                print(f"\n🎯 Quality Scores:")
                for metric, score in result.quality_scores.items():
                    print(f"   {metric}: {score:.2f}")
            else:
                print("\n❌ Quality scores not available")
            
            # Check if cost is available
            if hasattr(result, 'cost'):
                print(f"💰 Cost: ${result.cost:.4f}")
            else:
                print("❌ Cost not available")
                
        else:
            print("❌ No results returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_quality_scorer_directly():
    """Test the quality scorer directly."""
    
    print("\n🔧 Testing Quality Scorer Directly")
    print("-" * 60)
    
    try:
        from llm_eval_kit.metrics.quality_scorer import QualityScorer, calculate_cost
        
        # Test quality scoring
        scorer = QualityScorer()
        
        test_prompt = "我的订单为什么还没有发货？"
        test_response = """您的订单为什么还没有发货，可能有以下几种原因：

### 一、订单状态检查
1. **是否已付款？** - 请确认支付是否成功
2. **库存情况** - 商品可能暂时缺货

### 二、物流处理时间
- 通常24小时内处理
- 周末可能延迟

建议联系客服查询具体状态。"""
        
        scores = scorer.score_response(test_prompt, test_response, ["订单", "发货", "客服"])
        
        print("✅ Quality Scorer Test Results:")
        for metric, score in scores.items():
            print(f"   {metric}: {score:.2f}")
        
        # Test cost calculation
        cost = calculate_cost(500, "qwen-turbo")
        print(f"\n💰 Cost Calculation Test:")
        print(f"   500 tokens with qwen-turbo: ${cost:.4f}")
        
    except Exception as e:
        print(f"❌ Error testing quality scorer: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests."""
    
    # Test quality scorer directly first
    await test_quality_scorer_directly()
    
    # Then test the full evaluation pipeline
    await test_quality_scoring()
    
    print("\n" + "=" * 60)
    print("🎯 Quality Scoring Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
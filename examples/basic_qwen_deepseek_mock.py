"""
Mock demo for llm-eval-kit with Qwen and DeepSeek models.

This example demonstrates the toolkit's capability with Chinese LLM providers
using mock responses, so no real API keys are required.
"""

import asyncio
import os
from unittest.mock import Mock, patch
from dotenv import load_dotenv

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation

# Load environment variables from .env file
load_dotenv()


async def evaluate_model_mock(model_name, base_url, description):
    """Evaluate a specific model using mock responses."""
    
    model_config = {
        "base_url": base_url,
        "api_key": f"mock-{model_name.lower()}-key",
        "model": model_name,
        "timeout": 30
    }
    
    # Path to the sample data
    data_path = "examples/sample_data.jsonl"
    
    print(f"\n🚀 Evaluating {description} ({model_name})...")
    print(f"Base URL: {base_url}")
    print(f"Data: {data_path}")
    print("-" * 50)
    
    try:
        # Mock the OpenAI client to avoid real API calls
        with patch('llm_eval_kit.adapters.openai_compatible.OpenAI') as mock_openai:
            mock_client = Mock()
            
            # Create realistic mock responses for Chinese LLMs
            mock_responses = []
            
            # Sample 1: 订单查询
            mock_choice1 = Mock()
            mock_choice1.message.content = "请提供您的订单号，我将立即为您查询发货状态。我们通常会在24小时内处理订单。"
            mock_usage1 = Mock()
            mock_usage1.prompt_tokens = 18
            mock_usage1.completion_tokens = 25
            mock_usage1.total_tokens = 43
            mock_response1 = Mock()
            mock_response1.choices = [mock_choice1]
            mock_response1.usage = mock_usage1
            mock_responses.append(mock_response1)
            
            # Sample 2: 退货咨询
            mock_choice2 = Mock()
            mock_choice2.message.content = "我们支持7天无理由退货。如果产品有质量问题，请联系客服并提供订单详情，我们将为您处理退货事宜。"
            mock_usage2 = Mock()
            mock_usage2.prompt_tokens = 15
            mock_usage2.completion_tokens = 35
            mock_usage2.total_tokens = 50
            mock_response2 = Mock()
            mock_response2.choices = [mock_choice2]
            mock_response2.usage = mock_usage2
            mock_responses.append(mock_response2)
            
            # Set up mock responses
            mock_client.chat.completions.create.side_effect = mock_responses
            mock_openai.return_value = mock_client
            
            # Mock time for latency measurement
            with patch('llm_eval_kit.adapters.openai_compatible.time') as mock_time:
                # Simulate different response times for different models
                if "qwen" in model_name.lower():
                    time_values = [100.0, 101.5, 102.0, 102.8]  # Qwen is faster
                else:
                    time_values = [100.0, 101.8, 102.5, 103.3]  # DeepSeek is slightly slower
                mock_time.time.side_effect = time_values
                
                # Run the evaluation
                results = await run_evaluation(
                    data_path=data_path,
                    model_config=model_config,
                    verbose=True,  # Show detailed output
                    max_samples=2   # Limit to 2 samples for demo
                )
                
                print(f"✅ {model_name}: Mock evaluation completed successfully!")
                print(f"Processed {len(results)} samples")
                
                # Calculate average metrics
                if results:
                    avg_latency = sum(r.latency for r in results) / len(results)
                    total_tokens = sum(r.token_usage.get('total_tokens', 0) for r in results)
                    print(f"Average Latency: {avg_latency:.2f}s")
                    print(f"Total Tokens: {total_tokens}")
                
                return results
                
    except Exception as e:
        print(f"❌ {model_name}: Evaluation failed: {e}")
        return None


async def main():
    """Run mock evaluations with Qwen and DeepSeek models."""
    
    print("🚀 LLM-Eval-Kit: Qwen & DeepSeek Mock Evaluation")
    print("=" * 60)
    print("📝 This is a DEMO using mock responses")
    print("💡 No real API keys required")
    print("=" * 60)
    
    # Model configurations for Chinese LLM providers
    model_configs = [
        {
            "name": "Qwen-7B-Chat",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "description": "阿里云通义千问7B对话模型"
        },
        {
            "name": "DeepSeek-Chat",
            "base_url": "https://api.deepseek.com/v1",
            "description": "DeepSeek最新对话模型"
        },
        {
            "name": "Qwen-Turbo",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", 
            "description": "通义千问Turbo高速版本"
        },
        {
            "name": "DeepSeek-Coder",
            "base_url": "https://api.deepseek.com/v1",
            "description": "DeepSeek专业代码模型"
        }
    ]
    
    all_results = {}
    
    for config in model_configs:
        results = await evaluate_model_mock(
            model_name=config["name"],
            base_url=config["base_url"],
            description=config["description"]
        )
        
        if results:
            all_results[config["name"]] = {
                "results": results,
                "description": config["description"]
            }
    
    # Print summary comparison
    if all_results:
        print("\n" + "=" * 60)
        print("📊 Mock Evaluation Summary")
        print("=" * 60)
        
        for model_name, data in all_results.items():
            results = data["results"]
            description = data["description"]
            
            if results:
                avg_latency = sum(r.latency for r in results) / len(results)
                total_tokens = sum(r.token_usage.get('total_tokens', 0) for r in results)
                avg_response_length = sum(len(r.response) for r in results) / len(results)
                
                print(f"\n{description}:")
                print(f"  ✅ Samples: {len(results)}")
                print(f"  ⏱️  Avg Latency: {avg_latency:.2f}s")
                print(f"  📝 Total Tokens: {total_tokens}")
                print(f"  📏 Avg Response Length: {avg_response_length:.0f} chars")
                
                # Show a sample response
                if results:
                    sample = results[0]
                    print(f"  💬 Sample Response: {sample.response[:60]}...")
    
    print("\n" + "=" * 60)
    print("🎉 Mock Demo Completed!")
    print("\n💡 To use with real API keys:")
    print("1. Get Qwen API key from: https://dashscope.aliyun.com/")
    print("2. Get DeepSeek API key from: https://platform.deepseek.com/")
    print("3. Set QWEN_API_KEY and DEEPSEEK_API_KEY in .env file")
    print("4. Run: python examples/basic_qwen_deepseek.py")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
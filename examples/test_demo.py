"""
Test demo for llm-eval-kit without requiring real API keys.

This demo shows the basic functionality without making actual API calls.
"""

import asyncio
from unittest.mock import Mock, patch

# Import the evaluation toolkit
from llm_eval_kit.dataset.loader import DatasetLoader, EvaluationSample
from llm_eval_kit.reporter.console import ConsoleReporter, EvaluationResult


async def mock_evaluation_demo():
    """Run a mock evaluation demo."""
    
    print("🚀 LLM-Eval-Kit Demo")
    print("=" * 50)
    
    # Test dataset loading
    print("\n1. Testing dataset loading...")
    try:
        loader = DatasetLoader("examples/sample_data.jsonl")
        samples = loader.load()
        print(f"✅ Loaded {len(samples)} samples successfully")
        
        # Show first sample
        if samples:
            sample = samples[0]
            print(f"   Sample 1: ID={sample.id}, Question='{sample.question}'")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # Test console reporter
    print("\n2. Testing console reporter...")
    try:
        reporter = ConsoleReporter(verbose=True)
        
        # Create mock results
        mock_results = [
            EvaluationResult(
                sample_id="001",
                question="我的订单为什么还没有发货？",
                response="请提供订单号，我帮您查询发货状态",
                latency=1.2,
                token_usage={"total_tokens": 25, "prompt_tokens": 10, "completion_tokens": 15},
                model="gpt-3.5-turbo"
            ),
            EvaluationResult(
                sample_id="002", 
                question="产品有质量问题可以退货吗？",
                response="支持7天无理由退货，请提供订单详情",
                latency=0.8,
                token_usage={"total_tokens": 20, "prompt_tokens": 8, "completion_tokens": 12},
                model="gpt-3.5-turbo"
            )
        ]
        
        reporter.print_results(mock_results)
        print("✅ Console reporter working correctly")
        
    except Exception as e:
        print(f"❌ Console reporter failed: {e}")
        return
    
    # Test with mock OpenAI adapter
    print("\n3. Testing mock OpenAI adapter...")
    try:
        from llm_eval_kit.adapters.openai_compatible import OpenAICompatibleAdapter
        from llm_eval_kit.adapters.base import ModelResponse
        
        # Mock configuration
        config = {
            "api_key": "mock-key",
            "model": "gpt-3.5-turbo",
            "base_url": "https://api.example.com/v1"
        }
        
        # Mock the OpenAI client
        with patch('llm_eval_kit.adapters.openai_compatible.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Mock response from LLM"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 15
            
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Mock time for latency measurement
            with patch('llm_eval_kit.adapters.openai_compatible.time') as mock_time:
                mock_time.time.side_effect = [100.0, 101.2]
                
                adapter = OpenAICompatibleAdapter(config)
                result = await adapter.generate("Test question")
                
                print(f"✅ Mock adapter working:")
                print(f"   Response: {result.text}")
                print(f"   Latency: {result.latency:.2f}s")
                print(f"   Tokens: {result.token_usage['total_tokens']}")
                
    except Exception as e:
        print(f"❌ Mock adapter test failed: {e}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Set your OpenAI API key in .env file")
    print("2. Run: python examples/basic.py")
    print("3. Check the GitHub repo: https://github.com/xiaoKK903/LLM-Eval-Kit")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(mock_evaluation_demo())
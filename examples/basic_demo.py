"""
Basic demo for llm-eval-kit with mock API calls.

This version uses mock responses to demonstrate the functionality
without requiring real API keys.
"""

import asyncio
import os
from unittest.mock import Mock, patch

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation


async def main():
    """Run a basic evaluation demo with mock API calls."""
    
    print("🚀 LLM-Eval-Kit Basic Demo (Mock Mode)")
    print("=" * 60)
    
    # Configuration for the model (using mock)
    model_config = {
        "base_url": "https://api.openai.com/v1",
        "api_key": "mock-api-key-for-demo",
        "model": "gpt-3.5-turbo",
        "timeout": 30
    }
    
    # Path to the sample data
    data_path = "examples/sample_data.jsonl"
    
    print("Starting LLM evaluation in mock mode...")
    print(f"Data: {data_path}")
    print(f"Model: {model_config['model']}")
    print("-" * 60)
    
    try:
        # Mock the OpenAI client to avoid real API calls
        with patch('llm_eval_kit.adapters.openai_compatible.OpenAI') as mock_openai:
            mock_client = Mock()
            
            # Create proper mock responses for each sample
            mock_responses = []
            
            # Sample 1
            mock_choice1 = Mock()
            mock_choice1.message.content = "请提供订单号，我帮您查询发货状态"
            
            mock_usage1 = Mock()
            mock_usage1.prompt_tokens = 15
            mock_usage1.completion_tokens = 10
            mock_usage1.total_tokens = 25
            
            mock_response1 = Mock()
            mock_response1.choices = [mock_choice1]
            mock_response1.usage = mock_usage1
            mock_responses.append(mock_response1)
            
            # Sample 2
            mock_choice2 = Mock()
            mock_choice2.message.content = "支持7天无理由退货，请提供订单详情"
            
            mock_usage2 = Mock()
            mock_usage2.prompt_tokens = 12
            mock_usage2.completion_tokens = 8
            mock_usage2.total_tokens = 20
            
            mock_response2 = Mock()
            mock_response2.choices = [mock_choice2]
            mock_response2.usage = mock_usage2
            mock_responses.append(mock_response2)
            
            # Sample 3
            mock_choice3 = Mock()
            mock_choice3.message.content = "在个人中心-地址管理中可以修改收货地址"
            
            mock_usage3 = Mock()
            mock_usage3.prompt_tokens = 18
            mock_usage3.completion_tokens = 12
            mock_usage3.total_tokens = 30
            
            mock_response3 = Mock()
            mock_response3.choices = [mock_choice3]
            mock_response3.usage = mock_usage3
            mock_responses.append(mock_response3)
            
            # Set up mock responses
            mock_client.chat.completions.create.side_effect = mock_responses
            mock_openai.return_value = mock_client
            
            # Mock time for latency measurement
            with patch('llm_eval_kit.adapters.openai_compatible.time') as mock_time:
                # Simulate increasing time for each call
                time_values = [100.0, 101.2, 102.0, 102.8, 103.5]
                mock_time.time.side_effect = time_values
                
                # Run the evaluation
                results = await run_evaluation(
                    data_path=data_path,
                    model_config=model_config,
                    verbose=True,  # Show detailed output
                    max_samples=3   # Limit to 3 samples for demo
                )
                
                print("\n✅ Evaluation completed successfully!")
                print(f"Processed {len(results)} samples")
                
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        
        # Provide helpful error information
        if "OPENAI_API_KEY" in str(e):
            print("\n💡 To use real API calls:")
            print("1. Set your OpenAI API key in the .env file")
            print("2. Replace 'your_openai_api_key_here' with your actual key")
            print("3. Run: python examples/basic.py")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
"""
Quick model testing for llm-eval-kit.

This example quickly tests the most important models from Qwen and DeepSeek
to provide a fast comparison.
"""

import asyncio
import os
from dotenv import load_dotenv

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation

# Load environment variables from .env file
load_dotenv()


async def quick_evaluate_model(model_name, base_url, api_key_env_var, description):
    """Quickly evaluate a specific model."""
    
    api_key = os.getenv(api_key_env_var)
    
    if not api_key:
        return None
    
    # Special timeout for qwen-max which is slower
    timeout = 120 if "qwen-max" in model_name.lower() else 30
    
    model_config = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model_name,
        "timeout": timeout
    }
    
    # Path to the sample data
    data_path = "examples/sample_data.jsonl"
    
    try:
        # Run quick evaluation
        results = await run_evaluation(
            data_path=data_path,
            model_config=model_config,
            verbose=False,
            max_samples=1
        )
        
        if results:
            result = results[0]
            return {
                "model": model_name,
                "description": description,
                "latency": result.latency,
                "tokens": result.token_usage.get('total_tokens', 0),
                "response_length": len(result.response),
                "success": True
            }
        
    except Exception:
        pass
    
    return None


async def main():
    """Run quick evaluation of key models."""
    
    print("🚀 LLM-Eval-Kit: Quick Model Evaluation")
    print("=" * 60)
    
    # Key models to test (most important ones)
    key_models = [
        # Qwen Core Models
        {"name": "qwen-turbo", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "QWEN_API_KEY", "description": "千问Turbo"},
        {"name": "qwen-plus", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "QWEN_API_KEY", "description": "千问Plus"},
        {"name": "qwen-max", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "QWEN_API_KEY", "description": "千问Max"},
        
        # DeepSeek Core Models
        {"name": "deepseek-chat", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "description": "DeepSeek对话"},
        {"name": "deepseek-coder", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "description": "DeepSeek代码"},
        {"name": "deepseek-v4-pro", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "description": "DeepSeek V4 Pro"},
    ]
    
    print(f"Quick testing {len(key_models)} key models...")
    print("-" * 60)
    
    # Test all models concurrently for speed
    tasks = []
    for config in key_models:
        task = quick_evaluate_model(
            model_name=config["name"],
            base_url=config["base_url"],
            api_key_env_var=config["api_key_env"],
            description=config["description"]
        )
        tasks.append(task)
    
    # Run all tests concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    successful_results = []
    for i, result in enumerate(results):
        if result and not isinstance(result, Exception):
            successful_results.append(result)
            print(f"✅ {key_models[i]['description']:15} - {result['latency']:.2f}s - {result['tokens']} tokens")
        else:
            print(f"❌ {key_models[i]['description']:15} - Failed")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 QUICK EVALUATION SUMMARY")
    print("=" * 60)
    
    if successful_results:
        print(f"\n🏆 Performance Ranking:")
        
        # Sort by latency
        successful_results.sort(key=lambda x: x["latency"])
        
        for i, result in enumerate(successful_results, 1):
            print(f"   {i}. {result['description']:15} - {result['latency']:.2f}s - {result['tokens']} tokens")
        
        # Analysis
        fastest = successful_results[0]
        slowest = successful_results[-1]
        
        print(f"\n💡 Key Insights:")
        print(f"   • Fastest: {fastest['description']} ({fastest['latency']:.2f}s)")
        print(f"   • Slowest: {slowest['description']} ({slowest['latency']:.2f}s)")
        print(f"   • Speed range: {slowest['latency']/fastest['latency']:.1f}x difference")
        
        # Token efficiency
        token_efficient = min(successful_results, key=lambda x: x["tokens"])
        print(f"   • Most efficient: {token_efficient['description']} ({token_efficient['tokens']} tokens)")
    
    print(f"\n📈 Success Rate: {len(successful_results)}/{len(key_models)} models")
    
    print("\n🔧 Available Models Summary:")
    print("   • Qwen系列: Turbo(快), Plus(强), Max(最强)")
    print("   • DeepSeek: Chat(对话), Coder(代码), V4 Pro(最新)")
    print("   • 总计测试: 6个核心模型")


if __name__ == "__main__":
    # Run the quick evaluation
    asyncio.run(main())
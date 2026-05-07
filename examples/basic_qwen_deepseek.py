"""
Basic usage example for llm-eval-kit with Qwen and DeepSeek models.

This example demonstrates how to use the evaluation toolkit with
Chinese LLM providers like Qwen (Aliyun) and DeepSeek.
"""

import asyncio
import os
from dotenv import load_dotenv

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation

# Load environment variables from .env file
load_dotenv()


async def evaluate_model(model_name, base_url, api_key_env_var):
    """Evaluate a specific model."""
    
    api_key = os.getenv(api_key_env_var)
    
    if not api_key:
        print(f"❌ {model_name}: {api_key_env_var} environment variable not set")
        return None
    
    model_config = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model_name,
        "timeout": 30
    }
    
    # Path to the sample data
    data_path = "examples/sample_data.jsonl"
    
    print(f"\n🚀 Evaluating {model_name}...")
    print(f"Base URL: {base_url}")
    print(f"Data: {data_path}")
    print("-" * 50)
    
    try:
        # Run the evaluation
        results = await run_evaluation(
            data_path=data_path,
            model_config=model_config,
            verbose=True,  # Show detailed output
            max_samples=2   # Limit to 2 samples for demo
        )
        
        print(f"✅ {model_name}: Evaluation completed successfully!")
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
    """Run evaluations with Qwen and DeepSeek models."""
    
    print("🚀 LLM-Eval-Kit: Qwen & DeepSeek Evaluation")
    print("=" * 60)
    
    # Model configurations for Chinese LLM providers
    model_configs = [
        {
            "name": "qwen-turbo",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "QWEN_API_KEY",
            "description": "阿里云通义千问Turbo"
        },
        {
            "name": "qwen-plus",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key_env": "QWEN_API_KEY", 
            "description": "阿里云通义千问Plus"
        },
        {
            "name": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "description": "DeepSeek对话模型"
        },
        {
            "name": "deepseek-coder",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY",
            "description": "DeepSeek代码模型"
        }
    ]
    
    all_results = {}
    
    for config in model_configs:
        print(f"\n📋 {config['description']} ({config['name']})")
        
        results = await evaluate_model(
            model_name=config["name"],
            base_url=config["base_url"],
            api_key_env_var=config["api_key_env"]
        )
        
        if results:
            all_results[config["name"]] = {
                "results": results,
                "description": config["description"]
            }
    
    # Print summary comparison
    if all_results:
        print("\n" + "=" * 60)
        print("📊 Evaluation Summary")
        print("=" * 60)
        
        for model_name, data in all_results.items():
            results = data["results"]
            description = data["description"]
            
            if results:
                avg_latency = sum(r.latency for r in results) / len(results)
                total_tokens = sum(r.token_usage.get('total_tokens', 0) for r in results)
                
                print(f"\n{description} ({model_name}):")
                print(f"  ✅ Samples: {len(results)}")
                print(f"  ⏱️  Avg Latency: {avg_latency:.2f}s")
                print(f"  📝 Total Tokens: {total_tokens}")
    else:
        print("\n❌ No evaluations completed successfully.")
        print("\n💡 Setup Instructions:")
        print("1. Set QWEN_API_KEY in .env file for 千问 models")
        print("2. Set DEEPSEEK_API_KEY in .env file for DeepSeek models")
        print("3. Get API keys from respective provider dashboards")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
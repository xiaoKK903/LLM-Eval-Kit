"""
Comprehensive model testing for llm-eval-kit.

This example tests all available models from Qwen and DeepSeek to provide
a complete comparison of Chinese LLM capabilities.
"""

import asyncio
import os
from dotenv import load_dotenv

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation

# Load environment variables from .env file
load_dotenv()


async def evaluate_model(model_name, base_url, api_key_env_var, description):
    """Evaluate a specific model."""
    
    api_key = os.getenv(api_key_env_var)
    
    if not api_key:
        print(f"❌ {description}: {api_key_env_var} environment variable not set")
        return None
    
    model_config = {
        "base_url": base_url,
        "api_key": api_key,
        "model": model_name,
        "timeout": 60  # Longer timeout for comprehensive testing
    }
    
    # Path to the sample data
    data_path = "examples/sample_data.jsonl"
    
    print(f"\n🚀 Evaluating {description} ({model_name})...")
    print(f"Base URL: {base_url}")
    
    try:
        # Run the evaluation with a single sample for quick testing
        results = await run_evaluation(
            data_path=data_path,
            model_config=model_config,
            verbose=False,  # Less verbose for multiple models
            max_samples=1   # Test with 1 sample for efficiency
        )
        
        if results:
            result = results[0]
            avg_latency = result.latency
            total_tokens = result.token_usage.get('total_tokens', 0)
            response_length = len(result.response)
            
            print(f"✅ {description}: Success!")
            print(f"   ⏱️  Latency: {avg_latency:.2f}s")
            print(f"   📝 Tokens: {total_tokens}")
            print(f"   📏 Response: {response_length} chars")
            print(f"   💬 Sample: {result.response[:80]}...")
            
            return {
                "model": model_name,
                "description": description,
                "latency": avg_latency,
                "tokens": total_tokens,
                "response_length": response_length,
                "success": True
            }
        else:
            print(f"❌ {description}: No results returned")
            return {"model": model_name, "description": description, "success": False}
            
    except Exception as e:
        print(f"❌ {description}: Failed - {str(e)[:100]}...")
        return {"model": model_name, "description": description, "success": False, "error": str(e)}


async def main():
    """Run comprehensive evaluation of all available models."""
    
    print("🚀 LLM-Eval-Kit: Comprehensive Model Evaluation")
    print("=" * 70)
    
    # Comprehensive list of Chinese LLM models
    model_configs = [
        # Qwen Models (Aliyun)
        {"name": "qwen-turbo", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "QWEN_API_KEY", "description": "千问Turbo(高速版)"},
        {"name": "qwen-plus", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "QWEN_API_KEY", "description": "千问Plus(增强版)"},
        {"name": "qwen-max", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "QWEN_API_KEY", "description": "千问Max(最强版)"},
        {"name": "qwen-7b-chat", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "QWEN_API_KEY", "description": "千问7B对话"},
        {"name": "qwen-14b-chat", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "QWEN_API_KEY", "description": "千问14B对话"},
        
        # DeepSeek Models
        {"name": "deepseek-chat", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "description": "DeepSeek对话"},
        {"name": "deepseek-coder", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "description": "DeepSeek代码"},
        {"name": "deepseek-v4-pro", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "description": "DeepSeek V4 Pro"},
        {"name": "deepseek-v4-flash", "base_url": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY", "description": "DeepSeek V4 Flash"},
        
        # Additional Chinese LLMs (if available)
        # {"name": "baichuan-turbo", "base_url": "https://api.baichuan-ai.com/v1", "api_key_env": "BAICHUAN_API_KEY", "description": "百川Turbo"},
        # {"name": "chatglm-turbo", "base_url": "https://open.bigmodel.cn/api/paas/v4", "api_key_env": "CHATGLM_API_KEY", "description": "ChatGLM Turbo"},
    ]
    
    all_results = []
    successful_models = 0
    
    print(f"Testing {len(model_configs)} models...")
    print("-" * 70)
    
    for config in model_configs:
        result = await evaluate_model(
            model_name=config["name"],
            base_url=config["base_url"],
            api_key_env_var=config["api_key_env"],
            description=config["description"]
        )
        
        all_results.append(result)
        if result and result.get("success"):
            successful_models += 1
    
    # Print comprehensive summary
    print("\n" + "=" * 70)
    print("📊 COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📈 Overall Results:")
    print(f"   Total Models Tested: {len(model_configs)}")
    print(f"   Successful: {successful_models}")
    print(f"   Failed: {len(model_configs) - successful_models}")
    print(f"   Success Rate: {successful_models/len(model_configs)*100:.1f}%")
    
    # Successful models analysis
    successful_results = [r for r in all_results if r and r.get("success")]
    
    if successful_results:
        print(f"\n🏆 Performance Ranking (by Latency):")
        
        # Sort by latency (fastest first)
        successful_results.sort(key=lambda x: x["latency"])
        
        for i, result in enumerate(successful_results, 1):
            print(f"   {i:2d}. {result['description']:20} - {result['latency']:5.2f}s - {result['tokens']:4d} tokens")
        
        # Token efficiency analysis
        print(f"\n💡 Token Efficiency Analysis:")
        token_efficient = sorted(successful_results, key=lambda x: x["tokens"])
        
        for i, result in enumerate(token_efficient[:3], 1):
            print(f"   {i}. {result['description']:20} - {result['tokens']:4d} tokens (most efficient)")
        
        # Response quality analysis
        print(f"\n📝 Response Quality (by length):")
        detailed_responses = sorted(successful_results, key=lambda x: x["response_length"], reverse=True)
        
        for i, result in enumerate(detailed_responses[:3], 1):
            print(f"   {i}. {result['description']:20} - {result['response_length']:3d} chars (most detailed)")
    
    # Failed models analysis
    failed_results = [r for r in all_results if r and not r.get("success")]
    
    if failed_results:
        print(f"\n❌ Failed Models Analysis:")
        for result in failed_results:
            error_msg = result.get("error", "Unknown error")
            print(f"   {result['description']:20} - {error_msg[:60]}...")
    
    print("\n" + "=" * 70)
    print("🎯 RECOMMENDATIONS")
    print("=" * 70)
    
    if successful_results:
        fastest = successful_results[0]
        most_efficient = token_efficient[0] if token_efficient else fastest
        most_detailed = detailed_responses[0] if detailed_responses else fastest
        
        print(f"\n💡 Based on your use case:")
        print(f"   • For speed: {fastest['description']} ({fastest['latency']:.2f}s)")
        print(f"   • For cost efficiency: {most_efficient['description']} ({most_efficient['tokens']} tokens)")
        print(f"   • For detailed responses: {most_detailed['description']} ({most_detailed['response_length']} chars)")
    
    print(f"\n🔧 Next Steps:")
    print(f"   1. Run detailed evaluation: python examples/basic_qwen_deepseek.py")
    print(f"   2. Test with more samples for better statistics")
    print(f"   3. Compare models on your specific business data")


if __name__ == "__main__":
    # Run the comprehensive evaluation
    asyncio.run(main())
"""
Concurrent evaluation example for llm-eval-kit.

This example demonstrates the performance improvement from concurrent execution
compared to sequential execution.
"""

import asyncio
import time
import os
from dotenv import load_dotenv

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation

# Load environment variables
load_dotenv()


async def run_sequential_evaluation(model_configs, data_path, max_samples=2):
    """Run evaluations sequentially (one model at a time)."""
    
    print("🚀 Running Sequential Evaluation")
    print("=" * 60)
    
    start_time = time.time()
    all_results = {}
    
    for i, (model_name, config) in enumerate(model_configs.items(), 1):
        print(f"\n📊 Evaluating {model_name} ({i}/{len(model_configs)})...")
        
        try:
            # Run sequential evaluation (no concurrency)
            results = await run_evaluation(
                data_path=data_path,
                model_config=config,
                verbose=False,
                max_samples=max_samples,
                concurrency_per_model=1,  # No concurrency within model
                model_level_concurrency=1  # No concurrency between models
            )
            
            if results:
                avg_latency = sum(r.latency for r in results) / len(results)
                total_tokens = sum(r.token_usage.get('total_tokens', 0) for r in results)
                
                all_results[model_name] = {
                    'avg_latency': avg_latency,
                    'total_tokens': total_tokens,
                    'samples': len(results)
                }
                
                print(f"✅ {model_name}: {avg_latency:.2f}s, {total_tokens} tokens")
            else:
                print(f"❌ {model_name}: No results")
                
        except Exception as e:
            print(f"❌ {model_name}: Failed - {str(e)[:100]}")
    
    total_time = time.time() - start_time
    
    print(f"\n📈 Sequential Evaluation Summary:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Models Evaluated: {len(all_results)}")
    print(f"   Total Samples: {sum(r['samples'] for r in all_results.values())}")
    
    return all_results, total_time


async def run_concurrent_evaluation(model_configs, data_path, max_samples=2):
    """Run evaluations concurrently."""
    
    print("\n🚀 Running Concurrent Evaluation")
    print("=" * 60)
    
    start_time = time.time()
    all_results = {}
    
    # Create tasks for all models
    tasks = []
    for model_name, config in model_configs.items():
        task = run_evaluation(
            data_path=data_path,
            model_config=config,
            verbose=False,
            max_samples=max_samples,
            concurrency_per_model=3,  # Concurrent within model
            model_level_concurrency=4  # Concurrent between models
        )
        tasks.append((model_name, task))
    
    # Run all models concurrently
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
    
    # Process results
    for i, (model_name, _) in enumerate(tasks):
        result = results[i]
        
        if isinstance(result, Exception):
            print(f"❌ {model_name}: Failed - {str(result)[:100]}")
        else:
            if result:
                avg_latency = sum(r.latency for r in result) / len(result)
                total_tokens = sum(r.token_usage.get('total_tokens', 0) for r in result)
                
                all_results[model_name] = {
                    'avg_latency': avg_latency,
                    'total_tokens': total_tokens,
                    'samples': len(result)
                }
                
                print(f"✅ {model_name}: {avg_latency:.2f}s, {total_tokens} tokens")
            else:
                print(f"❌ {model_name}: No results")
    
    total_time = time.time() - start_time
    
    print(f"\n📈 Concurrent Evaluation Summary:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Models Evaluated: {len(all_results)}")
    print(f"   Total Samples: {sum(r['samples'] for r in all_results.values())}")
    
    return all_results, total_time


async def compare_performance():
    """Compare sequential vs concurrent performance."""
    
    # Model configurations for testing
    model_configs = {
        "qwen-turbo": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("QWEN_API_KEY"),
            "model": "qwen-turbo",
            "timeout": 30,
            "max_retries": 3
        },
        "qwen-plus": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.getenv("QWEN_API_KEY"),
            "model": "qwen-plus",
            "timeout": 30,
            "max_retries": 3
        },
        "deepseek-chat": {
            "base_url": "https://api.deepseek.com",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "model": "deepseek-chat",
            "timeout": 30,
            "max_retries": 3
        },
        "deepseek-coder": {
            "base_url": "https://api.deepseek.com",
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "model": "deepseek-coder",
            "timeout": 30,
            "max_retries": 3
        }
    }
    
    data_path = "examples/sample_data.jsonl"
    max_samples = 2
    
    print("🔬 LLM-Eval-Kit: Concurrent vs Sequential Performance Comparison")
    print("=" * 70)
    print(f"Testing {len(model_configs)} models with {max_samples} samples each")
    print("-" * 70)
    
    # Run sequential evaluation
    seq_results, seq_time = await run_sequential_evaluation(
        model_configs, data_path, max_samples
    )
    
    # Run concurrent evaluation
    conc_results, conc_time = await run_concurrent_evaluation(
        model_configs, data_path, max_samples
    )
    
    # Performance comparison
    print("\n" + "=" * 70)
    print("🏆 PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"\n📊 Execution Time Comparison:")
    print(f"   Sequential: {seq_time:.2f}s")
    print(f"   Concurrent: {conc_time:.2f}s")
    print(f"   Speedup: {seq_time / conc_time:.2f}x faster")
    
    print(f"\n📈 Throughput Comparison:")
    total_samples = sum(r['samples'] for r in seq_results.values())
    seq_throughput = total_samples / seq_time if seq_time > 0 else 0
    conc_throughput = total_samples / conc_time if conc_time > 0 else 0
    
    print(f"   Sequential: {seq_throughput:.2f} samples/second")
    print(f"   Concurrent: {conc_throughput:.2f} samples/second")
    print(f"   Throughput Improvement: {conc_throughput / seq_throughput:.2f}x")
    
    print(f"\n🔧 Model Performance (Concurrent Mode):")
    for model_name, result in conc_results.items():
        print(f"   {model_name}: {result['avg_latency']:.2f}s avg latency")
    
    print(f"\n💡 Key Insights:")
    print(f"   1. Concurrent execution significantly reduces total evaluation time")
    print(f"   2. Multiple models can be evaluated simultaneously")
    print(f"   3. Individual sample latencies remain similar")
    print(f"   4. Throughput increases dramatically with concurrency")


async def test_retry_mechanism():
    """Test the retry mechanism with simulated failures."""
    
    print("\n" + "=" * 70)
    print("🔧 Testing Retry Mechanism")
    print("=" * 70)
    
    # Test with a model that has aggressive timeout to trigger retries
    test_config = {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("QWEN_API_KEY"),
        "model": "qwen-turbo",
        "timeout": 5,  # Very short timeout to trigger retries
        "max_retries": 2,
        "retry_delay": 1
    }
    
    print("Testing retry mechanism with short timeout (5s)...")
    
    try:
        results = await run_evaluation(
            data_path="examples/sample_data.jsonl",
            model_config=test_config,
            verbose=True,
            max_samples=1
        )
        
        if results:
            print("✅ Retry test completed successfully!")
        else:
            print("❌ Retry test failed - no results")
            
    except Exception as e:
        print(f"❌ Retry test failed: {e}")


async def main():
    """Main function to run all tests."""
    
    # Check if API keys are available
    if not os.getenv("QWEN_API_KEY") or not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ API keys not found. Please set QWEN_API_KEY and DEEPSEEK_API_KEY in .env file")
        return
    
    # Run performance comparison
    await compare_performance()
    
    # Test retry mechanism (optional - can be commented out)
    # await test_retry_mechanism()
    
    print("\n" + "=" * 70)
    print("🎯 Day 2: Async Concurrency Optimization Complete!")
    print("=" * 70)
    
    print("\n✅ Implemented Features:")
    print("   1. Asynchronous concurrent execution")
    print("   2. Semaphore-based concurrency control")
    print("   3. Retry mechanism with exponential backoff")
    print("   4. Timeout control and error handling")
    print("   5. Performance monitoring and comparison")
    
    print("\n🚀 Next Steps:")
    print("   1. Test with larger datasets")
    print("   2. Optimize concurrency settings for specific APIs")
    print("   3. Add rate limiting for API providers")


if __name__ == "__main__":
    # Run the concurrent evaluation demo
    asyncio.run(main())
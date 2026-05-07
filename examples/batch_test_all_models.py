"""
Batch testing script for llm-eval-kit.

This script automatically tests all available models from Chinese LLM providers
and generates comprehensive comparison reports.
"""

import asyncio
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation

# Load environment variables
load_dotenv()

# Import the complete models list
from complete_models_list import get_recommended_models, get_all_models


class BatchTester:
    """Batch testing class for comprehensive model evaluation."""
    
    def __init__(self, max_samples=2, timeout=60):
        self.max_samples = max_samples
        self.timeout = timeout
        self.results = []
        self.start_time = None
        
    async def evaluate_model(self, model_config):
        """Evaluate a single model."""
        
        api_key = os.getenv(model_config["api_key_env"])
        if not api_key:
            return {
                "model": model_config["name"],
                "description": model_config["description"],
                "provider": model_config["provider"],
                "success": False,
                "error": f"API key not set: {model_config['api_key_env']}"
            }
        
        # Special timeout for slow models
        timeout = 120 if "qwen-max" in model_config["name"].lower() else self.timeout
        
        model_eval_config = {
            "base_url": model_config["base_url"],
            "api_key": api_key,
            "model": model_config["name"],
            "timeout": timeout
        }
        
        print(f"\n🚀 Testing {model_config['description']} ({model_config['name']})...")
        
        try:
            results = await run_evaluation(
                data_path="examples/sample_data.jsonl",
                model_config=model_eval_config,
                verbose=False,
                max_samples=self.max_samples
            )
            
            if results and len(results) > 0:
                # Calculate aggregate metrics
                total_latency = sum(r.latency for r in results)
                total_tokens = sum(r.token_usage.get('total_tokens', 0) for r in results)
                total_cost = sum(getattr(r, 'cost', 0) for r in results)
                avg_response_length = sum(len(r.response) for r in results) / len(results)
                
                # Calculate average quality scores
                avg_quality_scores = {}
                if results and hasattr(results[0], 'quality_scores') and results[0].quality_scores:
                    for metric in results[0].quality_scores.keys():
                        scores = [r.quality_scores.get(metric, 0) for r in results if hasattr(r, 'quality_scores') and r.quality_scores]
                        if scores:
                            avg_quality_scores[metric] = sum(scores) / len(scores)
                
                result = {
                    "model": model_config["name"],
                    "description": model_config["description"],
                    "provider": model_config["provider"],
                    "success": True,
                    "samples_tested": len(results),
                    "avg_latency": total_latency / len(results),
                    "total_tokens": total_tokens,
                    "total_cost": total_cost,
                    "avg_cost_per_sample": total_cost / len(results),
                    "avg_quality_scores": avg_quality_scores,
                    "avg_response_length": avg_response_length,
                    "timestamp": datetime.now().isoformat()
                }
                
                print(f"✅ {model_config['description']}: {result['avg_latency']:.2f}s")
                return result
            else:
                return {
                    "model": model_config["name"],
                    "description": model_config["description"],
                    "provider": model_config["provider"],
                    "success": False,
                    "error": "No results returned"
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ {model_config['description']}: {error_msg[:80]}...")
            return {
                "model": model_config["name"],
                "description": model_config["description"],
                "provider": model_config["provider"],
                "success": False,
                "error": error_msg
            }
    
    async def run_batch_test(self, models_to_test=None, test_mode="recommended"):
        """Run batch testing on specified models."""
        
        self.start_time = time.time()
        
        if models_to_test is None:
            if test_mode == "recommended":
                models_to_test = get_recommended_models()
            elif test_mode == "all":
                models_to_test = get_all_models()
            else:
                models_to_test = get_recommended_models()
        
        print("🚀 LLM-Eval-Kit: Batch Model Testing")
        print("=" * 70)
        print(f"📊 Test Mode: {test_mode}")
        print(f"📈 Models to test: {len(models_to_test)}")
        print(f"📝 Samples per model: {self.max_samples}")
        print("-" * 70)
        
        # Test models concurrently for efficiency
        tasks = []
        for model_config in models_to_test:
            task = self.evaluate_model(model_config)
            tasks.append(task)
        
        # Run all tests
        results = await asyncio.gather(*tasks)
        
        # Store results
        self.results = results
        
        # Generate report
        self.generate_report(test_mode)
        
        return results
    
    def generate_report(self, test_mode):
        """Generate comprehensive test report."""
        
        elapsed_time = time.time() - self.start_time
        successful_results = [r for r in self.results if r and r.get("success")]
        failed_results = [r for r in self.results if r and not r.get("success")]
        
        print("\n" + "=" * 70)
        print("📊 BATCH TESTING REPORT")
        print("=" * 70)
        
        print(f"\n📈 Overall Statistics:")
        print(f"   Total Models: {len(self.results)}")
        print(f"   Successful: {len(successful_results)}")
        print(f"   Failed: {len(failed_results)}")
        print(f"   Success Rate: {len(successful_results)/len(self.results)*100:.1f}%")
        print(f"   Total Time: {elapsed_time:.1f}s")
        
        if successful_results:
            # Performance ranking
            print(f"\n🏆 Performance Ranking (by Latency):")
            successful_results.sort(key=lambda x: x["avg_latency"])
            
            for i, result in enumerate(successful_results, 1):
                print(f"   {i:2d}. {result['description']:20} - {result['avg_latency']:6.2f}s - {result['total_tokens']:4d} tokens")
            
            # Provider comparison
            print(f"\n🏢 Provider Performance:")
            provider_stats = {}
            for result in successful_results:
                provider = result["provider"]
                if provider not in provider_stats:
                    provider_stats[provider] = {"count": 0, "total_latency": 0, "models": []}
                
                provider_stats[provider]["count"] += 1
                provider_stats[provider]["total_latency"] += result["avg_latency"]
                provider_stats[provider]["models"].append(result["description"])
            
            for provider, stats in provider_stats.items():
                avg_latency = stats["total_latency"] / stats["count"]
                print(f"   {provider:15} - {avg_latency:6.2f}s avg ({stats['count']} models)")
        
        if failed_results:
            print(f"\n❌ Failed Models Analysis:")
            for result in failed_results:
                error_msg = result.get("error", "Unknown error")
                print(f"   {result['description']:20} - {error_msg[:60]}...")
        
        # Save results to file
        self.save_results_to_file(test_mode)
        
        print(f"\n💡 Recommendations:")
        if successful_results:
            fastest = successful_results[0]
            most_efficient = min(successful_results, key=lambda x: x["total_tokens"])
            
            print(f"   • Fastest: {fastest['description']} ({fastest['avg_latency']:.2f}s)")
            print(f"   • Most Efficient: {most_efficient['description']} ({most_efficient['total_tokens']} tokens)")
        
        print(f"\n📁 Results saved to: results/batch_test_{test_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    def save_results_to_file(self, test_mode):
        """Save test results to JSON file."""
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/batch_test_{test_mode}_{timestamp}.json"
        
        report_data = {
            "test_mode": test_mode,
            "timestamp": datetime.now().isoformat(),
            "total_models": len(self.results),
            "successful_models": len([r for r in self.results if r and r.get("success")]),
            "results": self.results
        }
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return filename


async def main():
    """Main function for batch testing."""
    
    # Create batch tester
    tester = BatchTester(max_samples=2, timeout=60)
    
    # Test recommended models first
    print("🔧 Testing recommended models...")
    await tester.run_batch_test(test_mode="recommended")
    
    # Optionally test all models (commented out for now)
    # print("\n🔧 Testing all available models...")
    # await tester.run_batch_test(test_mode="all")


if __name__ == "__main__":
    # Run batch testing
    asyncio.run(main())
"""
Basic usage example for llm-eval-kit.

This example demonstrates the simplest way to use the evaluation toolkit.
"""

import asyncio
import os
from dotenv import load_dotenv

# Import the evaluation toolkit
from llm_eval_kit import run_evaluation

# Load environment variables from .env file
load_dotenv()


async def main():
    """Run a basic evaluation example."""
    
    # Configuration for the model
    # Note: You need to set OPENAI_API_KEY in your environment or .env file
    model_config = {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-3.5-turbo",
        "timeout": 30
    }
    
    # Check if API key is available
    if not model_config["api_key"]:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    
    # Path to the sample data
    data_path = "examples/sample_data.jsonl"
    
    print("Starting LLM evaluation...")
    print(f"Data: {data_path}")
    print(f"Model: {model_config['model']}")
    print("-" * 50)
    
    try:
        # Run the evaluation
        results = await run_evaluation(
            data_path=data_path,
            model_config=model_config,
            verbose=True,  # Show detailed output
            max_samples=3   # Limit to 3 samples for demo
        )
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
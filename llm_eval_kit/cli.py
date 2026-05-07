"""
CLI entry point for llm-eval-kit.
Usage: llm-eval-kit eval --model deepseek-chat --api-key xxx --data data.jsonl
"""

import argparse
import asyncio
import sys
import json

from .core.evaluator import Evaluator


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="llm-eval-kit",
        description="Lightweight LLM business evaluation toolkit",
    )
    sub = parser.add_subparsers(dest="command")

    # eval
    eval_p = sub.add_parser("eval", help="Run evaluation")
    eval_p.add_argument("--model", "-m", required=True, help="Model name")
    eval_p.add_argument("--api-key", "-k", required=True, help="API key")
    eval_p.add_argument("--base-url", "-u", default=None, help="API base URL")
    eval_p.add_argument("--data", "-d", required=True, help="Path to data.jsonl")
    eval_p.add_argument("--max-samples", "-n", type=int, default=None, help="Max samples")
    eval_p.add_argument("--concurrency", "-c", type=int, default=5, help="Concurrency")
    eval_p.add_argument("--output", "-o", default=None, help="Output path for JSON results")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command != "eval":
        print("Usage: llm-eval-kit eval --model NAME --api-key KEY --data PATH")
        sys.exit(1)

    model_cfg = {"model": args.model, "api_key": args.api_key}
    if args.base_url:
        model_cfg["base_url"] = args.base_url

    evaluator = Evaluator()
    result = asyncio.run(evaluator.run(
        models=[model_cfg],
        data_path=args.data,
        max_samples=args.max_samples,
        concurrency=args.concurrency,
        output_path=args.output,
    ))

    if args.output:
        print(f"Results saved to {args.output}")

    sys.exit(0)


if __name__ == "__main__":
    main()

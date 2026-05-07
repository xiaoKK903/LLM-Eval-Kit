"""
Configuration system for LLM-Eval-Kit.

Loads evaluation configuration from JSON files with
environment variable substitution.

Usage:
    from llm_eval_kit.utils.config import EvalConfigLoader

    config = EvalConfigLoader.from_file("config.json")
    # or
    config = EvalConfigLoader.from_dict({
        "models": [...],
        "data": "data.jsonl",
        "scorer": {"type": "rule"},
    })
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class ModelConfig:
    model: str
    api_key: str
    base_url: str
    name: Optional[str] = None

    def __post_init__(self):
        self.api_key = self._resolve_env(self.api_key)
        self.base_url = self._resolve_env(self.base_url)

    @staticmethod
    def _resolve_env(value: str) -> str:
        env_var = re.match(r'^\$\{(\w+)\}$', value)
        if env_var:
            env_name = env_var.group(1)
            env_value = os.environ.get(env_name)
            if env_value is None:
                raise ValueError(f"Environment variable {env_name} is not set")
            return env_value
        return value


@dataclass
class ScorerConfig:
    type: str = "rule"
    weights: Optional[Dict[str, float]] = None
    judge_model: Optional[str] = None


@dataclass
class OutputConfig:
    json: Optional[str] = None
    html: Optional[str] = None
    regression_baseline: Optional[str] = None


@dataclass
class EvalConfig:
    name: str = "unnamed evaluation"
    description: str = ""
    data_path: str = ""
    models: List[ModelConfig] = field(default_factory=list)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    max_samples: Optional[int] = None
    concurrency: int = 5

    def to_run_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "data_path": self.data_path,
            "concurrency": self.concurrency,
            "output_path": self.output.json,
        }
        if self.max_samples:
            kwargs["max_samples"] = self.max_samples
        return kwargs

    def to_model_configs(self) -> List[Dict[str, str]]:
        return [
            {
                "model": m.model,
                "api_key": m.api_key,
                "base_url": m.base_url,
            }
            for m in self.models
        ]


class EvalConfigLoader:
    @staticmethod
    def from_file(path: str) -> EvalConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EvalConfigLoader.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> EvalConfig:
        eval_data = data.get("evaluation", data)
        output_data = eval_data.get("output", {})

        scorer_data = eval_data.get("scorer", {})
        if isinstance(scorer_data, str):
            scorer_data = {"type": scorer_data}

        models = []
        for m in eval_data.get("models", []):
            models.append(ModelConfig(
                model=m["model"],
                api_key=m.get("api_key", "${API_KEY}"),
                base_url=m.get("base_url", ""),
                name=m.get("name"),
            ))

        return EvalConfig(
            name=eval_data.get("name", "unnamed evaluation"),
            description=eval_data.get("description", ""),
            data_path=eval_data.get("data") or eval_data.get("data_path", ""),
            models=models,
            scorer=ScorerConfig(
                type=scorer_data.get("type", "rule"),
                weights=scorer_data.get("weights"),
                judge_model=scorer_data.get("judge_model"),
            ),
            output=OutputConfig(
                json=output_data.get("json"),
                html=output_data.get("html"),
                regression_baseline=output_data.get("regression_baseline"),
            ),
            max_samples=eval_data.get("max_samples"),
            concurrency=eval_data.get("concurrency", 5),
        )

    @staticmethod
    def save_template(path: str):
        template = {
            "evaluation": {
                "name": "My Evaluation",
                "description": "Description of this evaluation",
                "data": "data.jsonl",
                "max_samples": 10,
                "concurrency": 5,
                "models": [
                    {
                        "model": "deepseek-chat",
                        "api_key": "${DEEPSEEK_API_KEY}",
                        "base_url": "https://api.deepseek.com/v1",
                    }
                ],
                "scorer": {
                    "type": "rule",
                },
                "output": {
                    "json": "results/report.json",
                    "html": "results/report.html",
                },
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        print(f"  Config template saved: {path}")

"""
Dataset loader for LLM evaluation datasets (JSONL format).
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class EvaluationSample:
    """A single evaluation sample."""
    id: str
    question: str
    reference: Optional[str] = None
    expected_keywords: Optional[List[str]] = None

    def to_dict(self) -> Dict:
        result = {"id": self.id, "question": self.question}
        if self.reference is not None:
            result["reference"] = self.reference
        if self.expected_keywords:
            result["expected_keywords"] = self.expected_keywords
        return result


class DatasetLoader:
    """Load evaluation datasets from JSONL files."""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._samples: List[EvaluationSample] = []

    def load(self) -> List[EvaluationSample]:
        self._samples = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        sample = self._parse_sample(data, line_num)
                        self._samples.append(sample)
                    except json.JSONDecodeError as e:
                        raise json.JSONDecodeError(
                            f"Invalid JSON on line {line_num}: {e}", e.doc, e.pos
                        )
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        return self._samples

    def validate(self):
        if not self._samples:
            raise ValueError(f"No samples loaded from {self.data_path}")

    def _parse_sample(self, data: Dict, line_num: int) -> EvaluationSample:
        if "id" not in data:
            raise ValueError(f"Missing 'id' field on line {line_num}")
        if "question" not in data:
            raise ValueError(f"Missing 'question' field on line {line_num}")
        if not isinstance(data["id"], str):
            raise ValueError(f"'id' must be string on line {line_num}")
        if not isinstance(data["question"], str):
            raise ValueError(f"'question' must be string on line {line_num}")

        return EvaluationSample(
            id=data["id"],
            question=data["question"],
            reference=data.get("reference"),
            expected_keywords=data.get("expected_keywords"),
        )

"""
Dataset loader for LLM evaluation.

This module provides functionality to load evaluation datasets from JSONL files.
Each line in the JSONL file should contain a sample with question and optional reference answer.
"""

import json
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass


@dataclass
class EvaluationSample:
    """Represents a single evaluation sample."""
    
    id: str
    question: str
    reference: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert sample to dictionary format."""
        result = {"id": self.id, "question": self.question}
        if self.reference is not None:
            result["reference"] = self.reference
        return result


class DatasetLoader:
    """Loader for evaluation datasets in JSONL format."""
    
    def __init__(self, data_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            data_path: Path to the JSONL file containing evaluation samples
        """
        self.data_path = data_path
        self._samples: List[EvaluationSample] = []
    
    def load(self) -> List[EvaluationSample]:
        """
        Load all samples from the dataset file.
        
        Returns:
            List of evaluation samples
            
        Raises:
            FileNotFoundError: If the data file does not exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
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
                            f"Invalid JSON on line {line_num}: {e}",
                            e.doc, e.pos
                        )
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        return self._samples
    
    def _parse_sample(self, data: Dict, line_num: int) -> EvaluationSample:
        """
        Parse a single sample from JSON data.
        
        Args:
            data: JSON data for a single sample
            line_num: Line number for error reporting
            
        Returns:
            Parsed evaluation sample
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate required fields
        if "id" not in data:
            raise ValueError(f"Missing 'id' field on line {line_num}")
        if "question" not in data:
            raise ValueError(f"Missing 'question' field on line {line_num}")
        
        # Validate field types
        if not isinstance(data["id"], str):
            raise ValueError(f"'id' must be string on line {line_num}")
        if not isinstance(data["question"], str):
            raise ValueError(f"'question' must be string on line {line_num}")
        
        # Parse optional reference field
        reference = None
        if "reference" in data:
            if not isinstance(data["reference"], str):
                raise ValueError(f"'reference' must be string on line {line_num}")
            reference = data["reference"]
        
        return EvaluationSample(
            id=data["id"],
            question=data["question"],
            reference=reference
        )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self._samples)
    
    def __iter__(self) -> Iterator[EvaluationSample]:
        """Iterate over samples in the dataset."""
        return iter(self._samples)
    
    def get_sample(self, index: int) -> EvaluationSample:
        """
        Get a specific sample by index.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            The evaluation sample at the specified index
            
        Raises:
            IndexError: If the index is out of range
        """
        if index < 0 or index >= len(self._samples):
            raise IndexError(f"Sample index {index} out of range")
        return self._samples[index]
    
    def validate(self) -> bool:
        """
        Validate that the dataset is properly loaded and all samples are valid.
        
        Returns:
            True if the dataset is valid, raises exception otherwise
        """
        if not self._samples:
            raise ValueError("Dataset is empty or not loaded")
        
        # Check for duplicate IDs
        ids = set()
        for sample in self._samples:
            if sample.id in ids:
                raise ValueError(f"Duplicate sample ID: {sample.id}")
            ids.add(sample.id)
        
        return True
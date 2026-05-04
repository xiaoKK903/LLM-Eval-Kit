"""
Basic tests for llm-eval-kit.

These tests verify the core functionality without making actual API calls.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from llm_eval_kit.dataset.loader import DatasetLoader, EvaluationSample
from llm_eval_kit.adapters.openai_compatible import OpenAICompatibleAdapter
from llm_eval_kit.reporter.console import ConsoleReporter, EvaluationResult


class TestDatasetLoader:
    """Test cases for dataset loading functionality."""
    
    def test_load_valid_jsonl(self):
        """Test loading a valid JSONL file."""
        # Create a temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "001", "question": "Test question 1"}\n')
            f.write('{"id": "002", "question": "Test question 2", "reference": "Test answer 2"}\n')
            temp_file = f.name
        
        try:
            # Load the dataset
            loader = DatasetLoader(temp_file)
            samples = loader.load()
            
            # Verify the results
            assert len(samples) == 2
            assert samples[0].id == "001"
            assert samples[0].question == "Test question 1"
            assert samples[0].reference is None
            
            assert samples[1].id == "002"
            assert samples[1].question == "Test question 2"
            assert samples[1].reference == "Test answer 2"
            
            # Test validation
            assert loader.validate() == True
            
        finally:
            # Clean up
            os.unlink(temp_file)
    
    def test_load_invalid_jsonl(self):
        """Test loading an invalid JSONL file."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "001", "question": "Test"}\n')
            f.write('invalid json line\n')
            temp_file = f.name
        
        try:
            loader = DatasetLoader(temp_file)
            with pytest.raises(json.JSONDecodeError):
                loader.load()
        finally:
            os.unlink(temp_file)
    
    def test_load_missing_required_fields(self):
        """Test loading JSONL with missing required fields."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"id": "001"}\n')  # Missing question field
            temp_file = f.name
        
        try:
            loader = DatasetLoader(temp_file)
            with pytest.raises(ValueError, match="Missing 'question' field"):
                loader.load()
        finally:
            os.unlink(temp_file)


class TestOpenAICompatibleAdapter:
    """Test cases for OpenAI-compatible adapter."""
    
    def test_valid_config(self):
        """Test adapter initialization with valid configuration."""
        config = {
            "api_key": "test-key",
            "model": "gpt-3.5-turbo",
            "base_url": "https://api.example.com/v1"
        }
        
        # Mock the OpenAI client to avoid compatibility issues
        with patch('llm_eval_kit.adapters.openai_compatible.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            adapter = OpenAICompatibleAdapter(config)
            assert adapter.model_name == "gpt-3.5-turbo"
    
    def test_invalid_config_missing_api_key(self):
        """Test adapter initialization with missing API key."""
        config = {
            "model": "gpt-3.5-turbo"
            # Missing api_key
        }
        
        with pytest.raises(ValueError, match="Missing required configuration field: api_key"):
            OpenAICompatibleAdapter(config)
    
    @pytest.mark.asyncio
    async def test_generate_method(self):
        """Test the generate method with mocked API call."""
        config = {
            "api_key": "test-key",
            "model": "gpt-3.5-turbo"
        }
        
        # Mock the OpenAI client and response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        
        with patch('llm_eval_kit.adapters.openai_compatible.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            # Mock time to control latency measurement
            with patch('llm_eval_kit.adapters.openai_compatible.time') as mock_time:
                mock_time.time.side_effect = [100.0, 101.5]  # start_time, end_time
                
                adapter = OpenAICompatibleAdapter(config)
                result = await adapter.generate("Test prompt")
                
                assert result.text == "Test response"
                assert result.latency == 1.5  # 101.5 - 100.0
                assert result.token_usage["total_tokens"] == 15


class TestConsoleReporter:
    """Test cases for console reporting functionality."""
    
    def test_truncate_text(self):
        """Test text truncation functionality."""
        reporter = ConsoleReporter()
        
        # Test short text (no truncation)
        short_text = "Hello"
        assert reporter._truncate_text(short_text, 10) == "Hello"
        
        # Test long text (truncation)
        long_text = "This is a very long text that needs to be truncated"
        truncated = reporter._truncate_text(long_text, 20)
        assert len(truncated) == 20
        assert truncated.endswith("...")
    
    def test_print_results_empty(self, capsys):
        """Test printing empty results."""
        reporter = ConsoleReporter()
        reporter.print_results([])
        
        captured = capsys.readouterr()
        assert "No results to display" in captured.out
    
    def test_print_results_with_data(self, capsys):
        """Test printing results with sample data."""
        results = [
            EvaluationResult(
                sample_id="001",
                question="Test question",
                response="Test response",
                latency=1.5,
                token_usage={"total_tokens": 20, "prompt_tokens": 10, "completion_tokens": 10},
                model="gpt-3.5-turbo"
            )
        ]
        
        reporter = ConsoleReporter(verbose=True)
        reporter.print_results(results)
        
        captured = capsys.readouterr()
        assert "LLM Evaluation Results" in captured.out
        assert "Test question" in captured.out
        assert "Test response" in captured.out


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
"""Test error handling and robustness of the email triage system."""

from unittest.mock import Mock, patch
from crewai_email_triage.pipeline import _triage_single, triage_email
from crewai_email_triage.classifier import ClassifierAgent
from crewai_email_triage.priority import PriorityAgent
from crewai_email_triage.summarizer import SummarizerAgent
from crewai_email_triage.response import ResponseAgent
from crewai_email_triage.config import load_config


class TestErrorHandling:
    """Test error handling in email processing."""

    def test_none_input_handling(self):
        """Test that None input is handled gracefully."""
        result = _triage_single(
            None,
            ClassifierAgent(),
            PriorityAgent(),
            SummarizerAgent(),
            ResponseAgent()
        )
        
        assert result["category"] == "unknown"
        assert result["priority"] == 0
        assert result["summary"] == "Processing failed"
        assert result["response"] == "Unable to process message"

    def test_empty_string_handling(self):
        """Test that empty string input is handled gracefully."""
        result = _triage_single(
            "",
            ClassifierAgent(),
            PriorityAgent(),
            SummarizerAgent(),
            ResponseAgent()
        )
        
        assert result["category"] == "empty"
        assert result["priority"] == 0
        assert result["summary"] == "Empty message"
        assert result["response"] == "No content to process"

    def test_whitespace_only_handling(self):
        """Test that whitespace-only input is handled gracefully."""
        result = _triage_single(
            "   \n\t  ",
            ClassifierAgent(),
            PriorityAgent(),
            SummarizerAgent(),
            ResponseAgent()
        )
        
        assert result["category"] == "empty"

    def test_agent_exception_handling(self):
        """Test that agent exceptions are caught and handled."""
        # Mock a failing classifier
        failing_classifier = Mock()
        failing_classifier.run.side_effect = Exception("Classifier failed")
        
        result = _triage_single(
            "test message",
            failing_classifier,
            PriorityAgent(),
            SummarizerAgent(),
            ResponseAgent()
        )
        
        assert result["category"] == "classification_error"
        assert isinstance(result["priority"], int)
        assert isinstance(result["summary"], str)
        assert isinstance(result["response"], str)

    def test_invalid_priority_score_handling(self):
        """Test that invalid priority scores are handled."""
        # Mock a priority agent that returns invalid data
        failing_prioritizer = Mock()
        failing_prioritizer.run.return_value = "priority: invalid"
        
        result = _triage_single(
            "test message",
            ClassifierAgent(),
            failing_prioritizer,
            SummarizerAgent(),
            ResponseAgent()
        )
        
        assert result["priority"] == 0

    def test_priority_score_range_validation(self):
        """Test that priority scores are capped to valid range."""
        # Mock a priority agent that returns out-of-range scores
        high_prioritizer = Mock()
        high_prioritizer.run.return_value = "priority: 99"
        
        result = _triage_single(
            "test message",
            ClassifierAgent(),
            high_prioritizer,
            SummarizerAgent(),
            ResponseAgent()
        )
        
        assert 0 <= result["priority"] <= 10

    def test_long_summary_truncation(self):
        """Test that very long summaries are truncated."""
        long_summarizer = Mock()
        long_summarizer.run.return_value = "summary: " + "x" * 600
        
        result = _triage_single(
            "test message",
            ClassifierAgent(),
            PriorityAgent(),
            long_summarizer,
            ResponseAgent()
        )
        
        assert len(result["summary"]) <= 500
        assert result["summary"].endswith("...")

    def test_long_response_truncation(self):
        """Test that very long responses are truncated."""
        long_responder = Mock()
        long_responder.run.return_value = "response: " + "x" * 1100
        
        result = _triage_single(
            "test message",
            ClassifierAgent(),
            PriorityAgent(),
            SummarizerAgent(),
            long_responder
        )
        
        assert len(result["response"]) <= 1000
        assert result["response"].endswith("...")

    def test_triage_email_error_handling(self):
        """Test that triage_email handles errors gracefully."""
        # This should not raise an exception even with problematic input
        result = triage_email(None)
        assert isinstance(result, dict)
        assert "category" in result
        assert "priority" in result
        assert "summary" in result
        assert "response" in result


class TestConfigErrorHandling:
    """Test configuration loading error handling."""

    def test_missing_config_file(self):
        """Test that missing config files are handled gracefully."""
        config = load_config("/nonexistent/path/config.json")
        
        # Should return fallback config
        assert isinstance(config, dict)
        assert "classifier" in config
        assert "priority" in config

    @patch("builtins.open", side_effect=OSError("File not accessible"))
    def test_config_file_access_error(self, mock_open):
        """Test that file access errors are handled."""
        config = load_config()
        
        assert isinstance(config, dict)
        assert "classifier" in config
        assert "priority" in config

    @patch("json.load", side_effect=ValueError("Invalid JSON"))
    def test_invalid_json_config(self, mock_json):
        """Test that invalid JSON configs are handled."""
        config = load_config()
        
        assert isinstance(config, dict)
        assert "classifier" in config
        assert "priority" in config

    def test_malformed_config_structure(self):
        """Test that configs with missing sections are handled."""
        with patch("json.load", return_value={"invalid": "structure"}):
            config = load_config()
            
            # Should merge with fallback
            assert "classifier" in config
            assert "priority" in config


class TestInputValidation:
    """Test input validation and sanitization."""

    def test_non_string_input(self):
        """Test that non-string inputs are handled."""
        for invalid_input in [123, [], {}, object()]:
            result = _triage_single(
                invalid_input,
                ClassifierAgent(),
                PriorityAgent(),
                SummarizerAgent(),
                ResponseAgent()
            )
            
            assert result["category"] == "unknown"
            assert result["priority"] == 0

    def test_unicode_handling(self):
        """Test that unicode content is handled properly."""
        unicode_content = "Hello 🌍 世界 café résumé"
        
        result = triage_email(unicode_content)
        
        # Should process without errors
        assert isinstance(result, dict)
        assert all(key in result for key in ["category", "priority", "summary", "response"])
"""Tests for pipeline method refactoring to ensure behavior is preserved."""

import pytest
from unittest.mock import Mock, patch
from crewai_email_triage.pipeline import _triage_single
from crewai_email_triage.agent_responses import (
    ClassificationResponse, PriorityResponse, 
    SummaryResponse, ResponseGenerationResponse
)
from crewai_email_triage.sanitization import SanitizationResult


class TestPipelineRefactoring:
    """Test suite for pipeline method refactoring - ensures behavior preservation."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        classifier = Mock()
        prioritizer = Mock()
        summarizer = Mock()
        responder = Mock()
        return classifier, prioritizer, summarizer, responder

    @pytest.fixture
    def valid_content(self):
        """Standard valid email content for testing."""
        return "Urgent: Please review the quarterly report by tomorrow morning."

    def test_input_validation_none_content(self, mock_agents):
        """Test that None content is handled correctly."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        result = _triage_single(None, classifier, prioritizer, summarizer, responder)
        
        assert result["category"] == "unknown"
        assert result["priority"] == 0
        assert result["summary"] == "Processing failed"
        assert result["response"] == "Unable to process message"

    def test_input_validation_empty_content(self, mock_agents):
        """Test that empty content is handled correctly."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        result = _triage_single("", classifier, prioritizer, summarizer, responder)
        
        assert result["category"] == "empty"
        assert result["priority"] == 0
        assert result["summary"] == "Empty message"
        assert result["response"] == "No content to process"

    def test_input_validation_whitespace_content(self, mock_agents):
        """Test that whitespace-only content is handled correctly."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        result = _triage_single("   \n\t  ", classifier, prioritizer, summarizer, responder)
        
        assert result["category"] == "empty"
        assert result["priority"] == 0
        assert result["summary"] == "Empty message"
        assert result["response"] == "No content to process"

    @patch('crewai_email_triage.pipeline.sanitize_email_content')
    def test_sanitization_success(self, mock_sanitize, mock_agents, valid_content):
        """Test successful content sanitization."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        # Mock successful sanitization
        sanitization_result = SanitizationResult(
            sanitized_content="Sanitized content",
            is_safe=True,
            threats_detected=[],
            modifications_made=[],
            original_length=len(valid_content),
            sanitized_length=16,
            processing_time_ms=5.0
        )
        mock_sanitize.return_value = sanitization_result
        
        # Mock successful agent responses
        with patch('crewai_email_triage.pipeline._run_agent_with_retry') as mock_retry, \
             patch('crewai_email_triage.pipeline.parse_agent_response') as mock_parse:
            
            mock_retry.return_value = "Success"
            mock_parse.return_value = ClassificationResponse(
                agent_type="classifier",
                success=True,
                raw_output="Success",
                category="urgent",
                processing_time_ms=10.0
            )
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Verify sanitization was called
            mock_sanitize.assert_called_once_with(valid_content)
            
            # Verify no sanitization warnings in clean content
            assert "sanitization_warnings" not in result

    @patch('crewai_email_triage.pipeline.sanitize_email_content')
    def test_sanitization_with_threats(self, mock_sanitize, mock_agents, valid_content):
        """Test sanitization when threats are detected."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        # Mock sanitization with threats
        sanitization_result = SanitizationResult(
            sanitized_content="Sanitized content",
            is_safe=False,
            threats_detected=["script_injection", "suspicious_url"],
            modifications_made=["removed_script", "blocked_url"],
            original_length=len(valid_content),
            sanitized_length=16,
            processing_time_ms=15.0
        )
        mock_sanitize.return_value = sanitization_result
        
        with patch('crewai_email_triage.pipeline._run_agent_with_retry') as mock_retry, \
             patch('crewai_email_triage.pipeline.parse_agent_response') as mock_parse:
            
            mock_retry.return_value = "Success"
            mock_parse.return_value = ClassificationResponse(
                agent_type="classifier",
                success=True,
                raw_output="Success",
                category="urgent",
                processing_time_ms=10.0
            )
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Verify sanitization warnings are included
            assert "sanitization_warnings" in result
            assert result["sanitization_warnings"] == ["script_injection", "suspicious_url"]

    @patch('crewai_email_triage.pipeline.sanitize_email_content')
    def test_sanitization_failure(self, mock_sanitize, mock_agents, valid_content):
        """Test handling of sanitization failures."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        # Mock sanitization failure
        mock_sanitize.side_effect = Exception("Sanitization error")
        
        with patch('crewai_email_triage.pipeline._run_agent_with_retry') as mock_retry, \
             patch('crewai_email_triage.pipeline.parse_agent_response') as mock_parse:
            
            mock_retry.return_value = "Success"
            mock_parse.return_value = ClassificationResponse(
                agent_type="classifier",
                success=True,
                raw_output="Success",
                category="urgent",
                processing_time_ms=10.0
            )
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Verify sanitization failure is handled
            assert "sanitization_warnings" in result
            assert result["sanitization_warnings"] == ["sanitization_failed"]

    @patch('crewai_email_triage.pipeline.sanitize_email_content')
    @patch('crewai_email_triage.pipeline._run_agent_with_retry')
    @patch('crewai_email_triage.pipeline.parse_agent_response')
    def test_classifier_success(self, mock_parse, mock_retry, mock_sanitize, mock_agents, valid_content):
        """Test successful classifier execution."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        # Mock sanitization success
        mock_sanitize.return_value = SanitizationResult(
            sanitized_content=valid_content, is_safe=True, threats_detected=[], 
            modifications_made=[], original_length=len(valid_content), 
            sanitized_length=len(valid_content), processing_time_ms=5.0
        )
        
        # Mock successful responses for all agents
        mock_retry.return_value = "Success"
        mock_parse.side_effect = [
            # Classifier response
            ClassificationResponse(agent_type="classifier", success=True, raw_output="urgent", category="urgent", processing_time_ms=10.0),
            # Priority response
            PriorityResponse(agent_type="priority", success=True, raw_output="8", priority_score=8, processing_time_ms=12.0),
            # Summarizer response
            SummaryResponse(agent_type="summarizer", success=True, raw_output="Test summary", summary="Test summary", processing_time_ms=15.0),
            # Responder response
            ResponseGenerationResponse(agent_type="responder", success=True, raw_output="Test response", response_text="Test response", processing_time_ms=20.0)
        ]
        
        result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
        
        assert result["category"] == "urgent"
        assert result["priority"] == 8
        assert result["summary"] == "Test summary"
        assert result["response"] == "Test response"

    @patch('crewai_email_triage.pipeline.sanitize_email_content')
    @patch('crewai_email_triage.pipeline._run_agent_with_retry')
    @patch('crewai_email_triage.pipeline.parse_agent_response')
    def test_classifier_parsing_failure(self, mock_parse, mock_retry, mock_sanitize, mock_agents, valid_content):
        """Test handling of classifier parsing failures."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        # Mock sanitization success
        mock_sanitize.return_value = SanitizationResult(
            sanitized_content=valid_content, is_safe=True, threats_detected=[], 
            modifications_made=[], original_length=len(valid_content), 
            sanitized_length=len(valid_content), processing_time_ms=5.0
        )
        
        mock_retry.return_value = "invalid_format"
        mock_parse.side_effect = [
            # Classifier parsing failure
            ClassificationResponse(agent_type="classifier", success=False, raw_output="invalid_format", error_message="Invalid format"),
            # Other agents succeed
            PriorityResponse(agent_type="priority", success=True, raw_output="5", priority_score=5, processing_time_ms=12.0),
            SummaryResponse(agent_type="summarizer", success=True, raw_output="Test summary", summary="Test summary", processing_time_ms=15.0),
            ResponseGenerationResponse(agent_type="responder", success=True, raw_output="Test response", response_text="Test response", processing_time_ms=20.0)
        ]
        
        result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
        
        assert result["category"] == "classification_error"
        assert result["priority"] == 5  # Other agents should still work
        assert result["summary"] == "Test summary"
        assert result["response"] == "Test response"

    @patch('crewai_email_triage.pipeline.sanitize_email_content')
    @patch('crewai_email_triage.pipeline._run_agent_with_retry')
    def test_classifier_exception(self, mock_retry, mock_sanitize, mock_agents, valid_content):
        """Test handling of classifier exceptions."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        # Mock sanitization success
        mock_sanitize.return_value = SanitizationResult(
            sanitized_content=valid_content, is_safe=True, threats_detected=[], 
            modifications_made=[], original_length=len(valid_content), 
            sanitized_length=len(valid_content), processing_time_ms=5.0
        )
        
        # Mock classifier exception, others succeed
        def side_effect(agent, content, agent_type):
            if agent_type == "classifier":
                raise Exception("Classifier error")
            return "Success"
        
        mock_retry.side_effect = side_effect
        
        with patch('crewai_email_triage.pipeline.parse_agent_response') as mock_parse, \
             patch('crewai_email_triage.pipeline._handle_agent_exception') as mock_handle:
            
            mock_handle.return_value = "exception_error"
            
            # Return appropriate response types for each agent
            def parse_side_effect(output, agent_type):
                if agent_type == "priority":
                    return PriorityResponse(
                        agent_type="priority", success=True, raw_output="5", priority_score=5, processing_time_ms=10.0
                    )
                elif agent_type == "summarizer":
                    return SummaryResponse(
                        agent_type="summarizer", success=True, raw_output="Summary", summary="Summary", processing_time_ms=10.0
                    )
                elif agent_type == "responder":
                    return ResponseGenerationResponse(
                        agent_type="responder", success=True, raw_output="Response", response_text="Response", processing_time_ms=10.0
                    )
                else:
                    return ClassificationResponse(
                        agent_type="classifier", success=True, raw_output="business", category="business", processing_time_ms=10.0
                    )
                    
            mock_parse.side_effect = parse_side_effect
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            assert result["category"] == "exception_error"
            # Should only handle the classifier exception, others succeed
            assert mock_handle.call_count == 1

    @patch('crewai_email_triage.pipeline.sanitize_email_content')
    def test_critical_pipeline_exception(self, mock_sanitize, mock_agents, valid_content):
        """Test handling of critical pipeline exceptions."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        # Mock sanitization failure that causes critical error
        mock_sanitize.side_effect = Exception("Critical error")
        
        with patch('crewai_email_triage.pipeline._run_agent_with_retry') as mock_retry:
            # Second exception in agent processing to trigger critical path
            mock_retry.side_effect = Exception("Another critical error")
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Should return default error values - actual implementation behavior
            assert result["category"] == "classification_error"
            assert result["priority"] == 0
            assert result["summary"] == "Summarization failed"
            assert result["response"] == "Response generation failed"

    def test_partial_agent_failures(self, mock_agents, valid_content):
        """Test that pipeline continues when some agents fail."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline.sanitize_email_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_agent_with_retry') as mock_retry, \
             patch('crewai_email_triage.pipeline.parse_agent_response') as mock_parse:
            
            # Mock sanitization success
            mock_sanitize.return_value = SanitizationResult(
                sanitized_content=valid_content, is_safe=True, threats_detected=[], 
                modifications_made=[], original_length=len(valid_content), 
                sanitized_length=len(valid_content), processing_time_ms=5.0
            )
            
            mock_retry.return_value = "Success"
            
            # Mock mixed success/failure responses
            mock_parse.side_effect = [
                # Classifier succeeds
                ClassificationResponse(agent_type="classifier", success=True, raw_output="urgent", category="urgent", processing_time_ms=10.0),
                # Priority fails
                PriorityResponse(agent_type="priority", success=False, raw_output="error", error_message="Priority error"),
                # Summarizer succeeds
                SummaryResponse(agent_type="summarizer", success=True, raw_output="Test summary", summary="Test summary", processing_time_ms=15.0),
                # Responder fails
                ResponseGenerationResponse(agent_type="responder", success=False, raw_output="error", error_message="Response error")
            ]
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Should have mixed results
            assert result["category"] == "urgent"  # Succeeded
            assert result["priority"] == 0  # Failed, default value
            assert result["summary"] == "Test summary"  # Succeeded
            assert result["response"] == "Response generation failed"  # Failed, default value

    def test_metrics_recording_throughout_pipeline(self, mock_agents, valid_content):
        """Test that metrics are recorded at each stage of the pipeline."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline.sanitize_email_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_agent_with_retry') as mock_retry, \
             patch('crewai_email_triage.pipeline.parse_agent_response') as mock_parse, \
             patch('crewai_email_triage.pipeline._metrics_collector') as mock_metrics:
            
            # Mock sanitization success
            mock_sanitize.return_value = SanitizationResult(
                sanitized_content=valid_content, is_safe=True, threats_detected=[], 
                modifications_made=[], original_length=len(valid_content), 
                sanitized_length=len(valid_content), processing_time_ms=5.0
            )
            
            mock_retry.return_value = "Success"
            
            # Return appropriate response types for each agent
            def parse_side_effect(output, agent_type):
                if agent_type == "classifier":
                    return ClassificationResponse(
                        agent_type="classifier", success=True, raw_output="Success", category="urgent", processing_time_ms=10.0
                    )
                elif agent_type == "priority":
                    return PriorityResponse(
                        agent_type="priority", success=True, raw_output="5", priority_score=5, processing_time_ms=10.0
                    )
                elif agent_type == "summarizer":
                    return SummaryResponse(
                        agent_type="summarizer", success=True, raw_output="Summary", summary="Summary", processing_time_ms=10.0
                    )
                else:  # responder
                    return ResponseGenerationResponse(
                        agent_type="responder", success=True, raw_output="Response", response_text="Response", processing_time_ms=10.0
                    )
                    
            mock_parse.side_effect = parse_side_effect
            
            _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Verify metrics were recorded for pipeline operations  
            assert mock_metrics.increment_counter.call_count >= 2  # At least some operations
            assert mock_metrics.record_histogram.call_count >= 2  # At least some timing metrics
"""Tests for graceful degradation improvements in the email triage pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from crewai_email_triage.pipeline import _triage_single
from crewai_email_triage.agent_responses import (
    ClassificationResponse, PriorityResponse, 
    SummaryResponse, ResponseGenerationResponse
)
from crewai_email_triage.sanitization import SanitizationResult


class TestGracefulDegradation:
    """Test suite for enhanced graceful degradation in the triage pipeline."""

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

    def test_sanitization_failure_allows_agents_to_run(self, mock_agents, valid_content):
        """Test that when sanitization fails, agents still run with original content."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline._sanitize_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_classifier') as mock_classifier, \
             patch('crewai_email_triage.pipeline._run_priority_agent') as mock_priority, \
             patch('crewai_email_triage.pipeline._run_summarizer') as mock_summarizer, \
             patch('crewai_email_triage.pipeline._run_responder') as mock_responder:
            
            # Mock sanitization failure
            mock_sanitize.side_effect = Exception("Critical sanitization error")
            
            # Mock successful agent operations that should modify the result
            def mock_classifier_success(agent, content, result):
                result["category"] = "urgent"
                
            def mock_priority_success(agent, content, result):
                result["priority"] = 8
                
            def mock_summarizer_success(agent, content, result):
                result["summary"] = "Test summary"
                
            def mock_responder_success(agent, content, result):
                result["response"] = "Test response"
            
            mock_classifier.side_effect = mock_classifier_success
            mock_priority.side_effect = mock_priority_success
            mock_summarizer.side_effect = mock_summarizer_success
            mock_responder.side_effect = mock_responder_success
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Should have results from all agents despite sanitization failure
            assert result["category"] == "urgent"
            assert result["priority"] == 8
            assert result["summary"] == "Test summary"
            assert result["response"] == "Test response"
            
            # Verify all agents were called
            mock_classifier.assert_called_once()
            mock_priority.assert_called_once()
            mock_summarizer.assert_called_once()
            mock_responder.assert_called_once()

    def test_individual_agent_critical_failure_isolation(self, mock_agents, valid_content):
        """Test that critical failure in one agent doesn't prevent others from running."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline._sanitize_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_classifier') as mock_classifier, \
             patch('crewai_email_triage.pipeline._run_priority_agent') as mock_priority, \
             patch('crewai_email_triage.pipeline._run_summarizer') as mock_summarizer, \
             patch('crewai_email_triage.pipeline._run_responder') as mock_responder:
            
            # Mock successful sanitization
            mock_sanitize.return_value = valid_content
            
            # Mock classifier critical failure, others succeed
            mock_classifier.side_effect = Exception("Critical classifier system error")
            
            def mock_priority_success(agent, content, result):
                result["priority"] = 8
                
            def mock_summarizer_success(agent, content, result):
                result["summary"] = "Test summary"
                
            def mock_responder_success(agent, content, result):
                result["response"] = "Test response"
            
            mock_priority.side_effect = mock_priority_success
            mock_summarizer.side_effect = mock_summarizer_success
            mock_responder.side_effect = mock_responder_success
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Should have default value for failed classifier but success for others
            assert result["category"] == "unknown"  # Default due to critical failure
            assert result["priority"] == 8  # Should succeed
            assert result["summary"] == "Test summary"  # Should succeed
            assert result["response"] == "Test response"  # Should succeed
            
            # Verify all agents were attempted
            mock_classifier.assert_called_once()
            mock_priority.assert_called_once()
            mock_summarizer.assert_called_once()
            mock_responder.assert_called_once()

    def test_multiple_agent_critical_failures_isolation(self, mock_agents, valid_content):
        """Test that multiple critical failures are isolated and don't cascade."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline._sanitize_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_classifier') as mock_classifier, \
             patch('crewai_email_triage.pipeline._run_priority_agent') as mock_priority, \
             patch('crewai_email_triage.pipeline._run_summarizer') as mock_summarizer, \
             patch('crewai_email_triage.pipeline._run_responder') as mock_responder:
            
            # Mock successful sanitization
            mock_sanitize.return_value = valid_content
            
            # Mock multiple critical failures
            mock_classifier.side_effect = Exception("Critical classifier error")
            mock_priority.side_effect = Exception("Critical priority error")
            
            # Mock successful agents
            def mock_summarizer_success(agent, content, result):
                result["summary"] = "Test summary"
                
            def mock_responder_success(agent, content, result):
                result["response"] = "Test response"
            
            mock_summarizer.side_effect = mock_summarizer_success
            mock_responder.side_effect = mock_responder_success
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Should have default values for failed agents but success for others
            assert result["category"] == "unknown"  # Default due to critical failure
            assert result["priority"] == 0  # Default due to critical failure
            assert result["summary"] == "Test summary"  # Should succeed
            assert result["response"] == "Test response"  # Should succeed
            
            # Verify all agents were attempted despite earlier failures
            mock_classifier.assert_called_once()
            mock_priority.assert_called_once()
            mock_summarizer.assert_called_once()
            mock_responder.assert_called_once()

    def test_all_agents_critical_failure_provides_defaults(self, mock_agents, valid_content):
        """Test that when all agents fail critically, default values are provided."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline._sanitize_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_classifier') as mock_classifier, \
             patch('crewai_email_triage.pipeline._run_priority_agent') as mock_priority, \
             patch('crewai_email_triage.pipeline._run_summarizer') as mock_summarizer, \
             patch('crewai_email_triage.pipeline._run_responder') as mock_responder:
            
            # Mock successful sanitization
            mock_sanitize.return_value = valid_content
            
            # Mock all agents failing critically
            mock_classifier.side_effect = Exception("Critical classifier error")
            mock_priority.side_effect = Exception("Critical priority error")
            mock_summarizer.side_effect = Exception("Critical summarizer error")
            mock_responder.side_effect = Exception("Critical responder error")
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Should have default values for all failed agents
            assert result["category"] == "unknown"
            assert result["priority"] == 0
            assert result["summary"] == "Processing failed"
            assert result["response"] == "Unable to process message"
            
            # Verify all agents were attempted
            mock_classifier.assert_called_once()
            mock_priority.assert_called_once()
            mock_summarizer.assert_called_once()
            mock_responder.assert_called_once()

    def test_sanitization_failure_with_mixed_agent_results(self, mock_agents, valid_content):
        """Test complex scenario: sanitization fails, some agents succeed, some fail."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline._sanitize_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_classifier') as mock_classifier, \
             patch('crewai_email_triage.pipeline._run_priority_agent') as mock_priority, \
             patch('crewai_email_triage.pipeline._run_summarizer') as mock_summarizer, \
             patch('crewai_email_triage.pipeline._run_responder') as mock_responder:
            
            # Mock sanitization failure
            mock_sanitize.side_effect = Exception("Sanitization failed")
            
            # Mock mixed agent results
            def mock_classifier_success(agent, content, result):
                result["category"] = "urgent"
                
            mock_classifier.side_effect = mock_classifier_success
            mock_priority.side_effect = Exception("Priority agent failed")
            
            def mock_summarizer_success(agent, content, result):
                result["summary"] = "Brief summary"
                
            mock_summarizer.side_effect = mock_summarizer_success
            mock_responder.side_effect = Exception("Responder agent failed")
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Should have mixed results
            assert result["category"] == "urgent"  # Succeeded
            assert result["priority"] == 0  # Failed, default value
            assert result["summary"] == "Brief summary"  # Succeeded
            assert result["response"] == "Unable to process message"  # Failed, default value
            
            # Should have sanitization warning
            assert "sanitization_warnings" in result
            assert result["sanitization_warnings"] == ["sanitization_failed"]

    def test_graceful_degradation_maintains_metrics(self, mock_agents, valid_content):
        """Test that graceful degradation properly maintains metrics for successful agents."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline._sanitize_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_classifier') as mock_classifier, \
             patch('crewai_email_triage.pipeline._run_priority_agent') as mock_priority, \
             patch('crewai_email_triage.pipeline._run_summarizer') as mock_summarizer, \
             patch('crewai_email_triage.pipeline._run_responder') as mock_responder, \
             patch('crewai_email_triage.pipeline._metrics_collector') as mock_metrics:
            
            # Mock successful sanitization
            mock_sanitize.return_value = valid_content
            
            # Mock some agents succeeding, some failing
            def mock_classifier_success(agent, content, result):
                result["category"] = "urgent"
                
            mock_classifier.side_effect = mock_classifier_success
            mock_priority.side_effect = Exception("Priority failed")
            
            def mock_summarizer_success(agent, content, result):
                result["summary"] = "Summary"
                
            mock_summarizer.side_effect = mock_summarizer_success
            mock_responder.side_effect = Exception("Responder failed")
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Verify the result has mixed success/failure
            assert result["category"] == "urgent"
            assert result["priority"] == 0
            assert result["summary"] == "Summary"  
            assert result["response"] == "Unable to process message"
            
            # Verify that metrics were still collected (the implementation should track attempts)
            # The exact calls depend on implementation, but metrics collection should continue
            assert mock_metrics.increment_counter.called

    def test_error_recovery_preserves_existing_results(self, mock_agents, valid_content):
        """Test that when agents update results progressively, earlier successes are preserved."""
        classifier, prioritizer, summarizer, responder = mock_agents
        
        with patch('crewai_email_triage.pipeline._sanitize_content') as mock_sanitize, \
             patch('crewai_email_triage.pipeline._run_classifier') as mock_classifier, \
             patch('crewai_email_triage.pipeline._run_priority_agent') as mock_priority, \
             patch('crewai_email_triage.pipeline._run_summarizer') as mock_summarizer, \
             patch('crewai_email_triage.pipeline._run_responder') as mock_responder:
            
            # Mock successful sanitization
            mock_sanitize.return_value = valid_content
            
            # Mock progressive success then failure
            def mock_classifier_success(agent, content, result):
                result["category"] = "urgent"
                
            def mock_priority_success(agent, content, result):
                result["priority"] = 9
                
            mock_classifier.side_effect = mock_classifier_success
            mock_priority.side_effect = mock_priority_success
            mock_summarizer.side_effect = Exception("Summarizer failed")
            mock_responder.side_effect = Exception("Responder failed")
            
            result = _triage_single(valid_content, classifier, prioritizer, summarizer, responder)
            
            # Should preserve earlier successes
            assert result["category"] == "urgent"  # From classifier
            assert result["priority"] == 9  # From priority
            assert result["summary"] == "Processing failed"  # Default due to failure
            assert result["response"] == "Unable to process message"  # Default due to failure
"""Tests for real-time intelligence system."""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from crewai_email_triage.realtime_intelligence import (
    EmailFlowEvent, FlowPriority, ProcessingStatus,
    IntelligentRouter, RealTimeProcessor, RealTimeIntelligenceManager,
    get_intelligence_manager, submit_email_for_processing
)


class TestEmailFlowEvent:
    """Test EmailFlowEvent data class."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        event = EmailFlowEvent(
            id="test_123",
            content="Test email content"
        )
        
        assert event.id == "test_123"
        assert event.content == "Test email content"
        assert event.priority == FlowPriority.MEDIUM
        assert event.status == ProcessingStatus.QUEUED
        assert event.retry_count == 0
        assert isinstance(event.timestamp, datetime)
    
    def test_event_with_custom_priority(self):
        """Test event creation with custom priority."""
        event = EmailFlowEvent(
            id="urgent_123",
            content="Urgent email!",
            priority=FlowPriority.CRITICAL
        )
        
        assert event.priority == FlowPriority.CRITICAL


class TestIntelligentRouter:
    """Test intelligent routing system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.router = IntelligentRouter()
    
    def test_priority_score_calculation(self):
        """Test priority score calculation."""
        event = EmailFlowEvent(
            id="test_1",
            content="Normal email",
            priority=FlowPriority.MEDIUM
        )
        
        score = self.router.calculate_priority_score(event)
        assert 0.0 < score <= 1.5
    
    def test_urgent_content_boost(self):
        """Test that urgent keywords boost priority."""
        normal_event = EmailFlowEvent(
            id="normal",
            content="Regular meeting update",
            priority=FlowPriority.MEDIUM
        )
        
        urgent_event = EmailFlowEvent(
            id="urgent",
            content="URGENT: System down!",
            priority=FlowPriority.MEDIUM
        )
        
        normal_score = self.router.calculate_priority_score(normal_event)
        urgent_score = self.router.calculate_priority_score(urgent_event)
        
        assert urgent_score > normal_score
    
    def test_time_factor_boost(self):
        """Test that older emails get priority boost."""
        old_timestamp = datetime.utcnow() - timedelta(hours=1)
        
        old_event = EmailFlowEvent(
            id="old",
            content="Old email",
            timestamp=old_timestamp
        )
        
        score = self.router.calculate_priority_score(old_event)
        assert score > 0.5  # Should get time boost
    
    def test_retry_penalty(self):
        """Test retry penalty reduces priority."""
        event = EmailFlowEvent(
            id="retry_test",
            content="Test email",
            retry_count=2
        )
        
        score = self.router.calculate_priority_score(event)
        
        # Reset retry count and compare
        event.retry_count = 0
        fresh_score = self.router.calculate_priority_score(event)
        
        assert score < fresh_score
    
    def test_route_event_high_priority(self):
        """Test routing of high priority events."""
        critical_event = EmailFlowEvent(
            id="critical",
            content="EMERGENCY: Data breach detected!",
            priority=FlowPriority.CRITICAL
        )
        
        queue = self.router.route_event(critical_event)
        assert queue == "high_priority_queue"
    
    def test_route_event_bulk(self):
        """Test routing of bulk events."""
        bulk_event = EmailFlowEvent(
            id="bulk",
            content="Newsletter update",
            priority=FlowPriority.BULK
        )
        
        queue = self.router.route_event(bulk_event)
        assert queue == "bulk_queue"


class TestRealTimeProcessor:
    """Test real-time processor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = RealTimeProcessor(max_workers=2, batch_size=5)
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.max_workers == 2
        assert self.processor.batch_size == 5
        assert not self.processor.running
        assert len(self.processor.queues) == 3
    
    def test_enqueue_event_success(self):
        """Test successful event enqueueing."""
        event = EmailFlowEvent(
            id="test_enqueue",
            content="Valid email content"
        )
        
        with patch('crewai_email_triage.realtime_intelligence.validate_email_content', return_value=True):
            result = self.processor.enqueue_event(event)
            assert result is True
    
    def test_enqueue_invalid_content(self):
        """Test enqueueing with invalid content."""
        event = EmailFlowEvent(
            id="invalid_content",
            content=""
        )
        
        with patch('crewai_email_triage.realtime_intelligence.validate_email_content', return_value=False):
            result = self.processor.enqueue_event(event)
            assert result is False
    
    @patch('crewai_email_triage.realtime_intelligence.triage_email')
    def test_process_single_event_success(self, mock_triage):
        """Test successful single event processing."""
        mock_triage.return_value = {"category": "test", "priority": 5}
        
        event = EmailFlowEvent(
            id="process_test",
            content="Test email"
        )
        
        result = self.processor.process_single_event(event)
        
        assert result is not None
        assert event.status == ProcessingStatus.COMPLETED
        assert event.processing_metadata["result"] == {"category": "test", "priority": 5}
        mock_triage.assert_called_once_with("Test email")
    
    @patch('crewai_email_triage.realtime_intelligence.triage_email')
    def test_process_single_event_failure(self, mock_triage):
        """Test event processing failure and retry logic."""
        mock_triage.side_effect = Exception("Processing failed")
        
        event = EmailFlowEvent(
            id="fail_test",
            content="Test email",
            max_retries=1
        )
        
        result = self.processor.process_single_event(event)
        
        assert result is None
        assert event.retry_count == 1
        assert event.status == ProcessingStatus.FAILED
        assert "Processing failed" in event.error_details
    
    def test_metrics_update(self):
        """Test metrics updating."""
        # Add some processing times
        self.processor.processing_times = [100.0, 150.0, 200.0]
        
        self.processor.update_metrics()
        
        assert self.processor.metrics.avg_processing_time_ms == 150.0
    
    def test_add_error_handler(self):
        """Test adding custom error handler."""
        handler_called = False
        
        def test_handler(event, error):
            nonlocal handler_called
            handler_called = True
        
        self.processor.add_error_handler(test_handler)
        assert len(self.processor.error_handlers) == 1
        
        # Trigger error handler
        with patch('crewai_email_triage.realtime_intelligence.triage_email', side_effect=Exception("Test error")):
            event = EmailFlowEvent(id="error_test", content="Test", max_retries=1)
            self.processor.process_single_event(event)
        
        assert handler_called
    
    def test_get_status(self):
        """Test status reporting."""
        status = self.processor.get_status()
        
        assert "running" in status
        assert "active_processors" in status
        assert "max_workers" in status
        assert "queue_depths" in status
        assert "metrics" in status
        
        assert status["max_workers"] == 2
        assert status["running"] is False


class TestRealTimeIntelligenceManager:
    """Test the main intelligence manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = RealTimeIntelligenceManager()
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        assert self.manager.processor is not None
        assert self.manager.event_counter == 0
    
    def test_submit_email_success(self):
        """Test successful email submission."""
        with patch.object(self.manager.processor, 'enqueue_event', return_value=True):
            event_id = self.manager.submit_email("Test email content")
            
            assert event_id.startswith("email_1_")
            assert self.manager.event_counter == 1
    
    def test_submit_email_failure(self):
        """Test email submission failure."""
        with patch.object(self.manager.processor, 'enqueue_event', return_value=False):
            with pytest.raises(ValueError):
                self.manager.submit_email("Invalid content")
    
    def test_submit_with_priority(self):
        """Test email submission with custom priority."""
        with patch.object(self.manager.processor, 'enqueue_event', return_value=True) as mock_enqueue:
            self.manager.submit_email(
                "Urgent email!",
                priority=FlowPriority.CRITICAL
            )
            
            # Verify the event was created with correct priority
            mock_enqueue.assert_called_once()
            event = mock_enqueue.call_args[0][0]
            assert event.priority == FlowPriority.CRITICAL
    
    @patch('crewai_email_triage.realtime_intelligence.get_health_checker')
    def test_get_system_status(self, mock_health_checker):
        """Test system status reporting."""
        # Mock health checker
        mock_health_result = Mock()
        mock_health_result.status.value = "healthy"
        mock_health_result.response_time_ms = 50.0
        mock_health_result.checks = [Mock(status=Mock(name="HEALTHY")) for _ in range(3)]
        
        mock_health_checker.return_value.check_health.return_value = mock_health_result
        
        status = self.manager.get_system_status()
        
        assert "realtime_processor" in status
        assert "system_health" in status
        assert "timestamp" in status
        assert "version" in status
        
        assert status["system_health"]["status"] == "healthy"
        assert status["system_health"]["response_time_ms"] == 50.0
        assert status["system_health"]["healthy_checks"] == 3


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_intelligence_manager_singleton(self):
        """Test that get_intelligence_manager returns same instance."""
        manager1 = get_intelligence_manager()
        manager2 = get_intelligence_manager()
        
        assert manager1 is manager2
    
    @patch('crewai_email_triage.realtime_intelligence.get_intelligence_manager')
    def test_submit_email_for_processing(self, mock_get_manager):
        """Test convenience function for email submission."""
        mock_manager = Mock()
        mock_manager.submit_email.return_value = "test_event_id"
        mock_get_manager.return_value = mock_manager
        
        event_id = submit_email_for_processing("Test email")
        
        assert event_id == "test_event_id"
        mock_manager.submit_email.assert_called_once_with(
            "Test email", None, FlowPriority.MEDIUM
        )


class TestIntegration:
    """Integration tests for the real-time intelligence system."""
    
    def test_full_workflow(self):
        """Test complete workflow from submission to processing."""
        manager = RealTimeIntelligenceManager()
        
        # Submit an email
        with patch.object(manager.processor, 'enqueue_event', return_value=True):
            event_id = manager.submit_email("Integration test email")
        
        # Check that event was created
        assert event_id.startswith("email_")
        
        # Get system status
        status = manager.get_system_status()
        assert status["realtime_processor"]["running"] is False
        assert status["version"] == "2.0.0"
    
    def test_error_handling_workflow(self):
        """Test error handling in the workflow."""
        processor = RealTimeProcessor(max_workers=1)
        
        # Add custom error handler
        errors_handled = []
        
        def error_handler(event, error):
            errors_handled.append((event.id, str(error)))
        
        processor.add_error_handler(error_handler)
        
        # Create failing event
        with patch('crewai_email_triage.realtime_intelligence.triage_email', side_effect=Exception("Test failure")):
            event = EmailFlowEvent(id="error_test", content="Test", max_retries=1)
            result = processor.process_single_event(event)
        
        assert result is None
        assert len(errors_handled) == 1
        assert errors_handled[0][0] == "error_test"
        assert "Test failure" in errors_handled[0][1]


@pytest.mark.performance
class TestPerformance:
    """Performance tests for real-time intelligence."""
    
    def test_high_throughput_processing(self):
        """Test system performance under high load."""
        processor = RealTimeProcessor(max_workers=4, batch_size=20)
        
        # Create multiple events
        events = [
            EmailFlowEvent(id=f"perf_test_{i}", content=f"Test email {i}")
            for i in range(100)
        ]
        
        start_time = time.time()
        
        # Enqueue all events
        with patch('crewai_email_triage.realtime_intelligence.validate_email_content', return_value=True):
            for event in events:
                processor.enqueue_event(event)
        
        enqueue_time = time.time() - start_time
        
        # Verify performance
        assert enqueue_time < 1.0  # Should enqueue 100 events in less than 1 second
        assert sum(q.qsize() for q in processor.queues.values()) == 100
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over time."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        processor = RealTimeProcessor()
        
        # Add and process many events
        for i in range(1000):
            processor.processing_times.append(float(i))
            if i % 100 == 0:
                processor.update_metrics()
        
        # Check memory hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / initial_memory
        
        assert memory_growth < 0.5  # Less than 50% memory growth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
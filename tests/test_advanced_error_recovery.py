"""Tests for advanced error recovery system."""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from crewai_email_triage.advanced_error_recovery import (
    FailureType, RecoveryAction, FailureEvent, RecoveryRule,
    ErrorPatternAnalyzer, SelfHealingOrchestrator,
    get_self_healing_orchestrator, handle_failure_with_recovery
)


class TestFailureEvent:
    """Test FailureEvent data class."""
    
    def test_failure_event_creation(self):
        """Test basic failure event creation."""
        event = FailureEvent(
            id="test_failure_123",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.TRANSIENT,
            component="test_component",
            error_message="Test error occurred"
        )
        
        assert event.id == "test_failure_123"
        assert event.failure_type == FailureType.TRANSIENT
        assert event.component == "test_component"
        assert event.error_message == "Test error occurred"
        assert not event.resolved
        assert event.resolution_time is None
        assert len(event.recovery_attempts) == 0


class TestErrorPatternAnalyzer:
    """Test error pattern analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ErrorPatternAnalyzer()
    
    def test_analyze_transient_failure(self):
        """Test classification of transient failures."""
        error = ConnectionError("Connection timeout occurred")
        context = {"component": "email_processor"}
        
        failure_event = self.analyzer.analyze_failure(error, "email_processor", context)
        
        assert failure_event.failure_type == FailureType.TRANSIENT
        assert failure_event.component == "email_processor"
        assert "Connection timeout" in failure_event.error_message
        assert len(self.analyzer.failure_history) == 1
    
    def test_analyze_resource_failure(self):
        """Test classification of resource failures."""
        error = MemoryError("Out of memory")
        context = {"memory_usage": "high"}
        
        failure_event = self.analyzer.analyze_failure(error, "processor", context)
        
        assert failure_event.failure_type == FailureType.RESOURCE
        assert failure_event.component == "processor"
    
    def test_analyze_security_failure(self):
        """Test classification of security failures."""
        error = PermissionError("Unauthorized access attempt")
        context = {"user": "unknown"}
        
        failure_event = self.analyzer.analyze_failure(error, "auth_service", context)
        
        assert failure_event.failure_type == FailureType.SECURITY
        assert failure_event.component == "auth_service"
    
    def test_analyze_configuration_failure(self):
        """Test classification of configuration failures."""
        error = ValueError("Invalid configuration key")
        context = {"config_file": "app.json"}
        
        failure_event = self.analyzer.analyze_failure(error, "config_loader", context)
        
        assert failure_event.failure_type == FailureType.CONFIGURATION
    
    def test_analyze_corruption_failure(self):
        """Test classification of data corruption failures."""
        error = ValueError("Malformed JSON data")
        context = {"data_source": "user_input"}
        
        failure_event = self.analyzer.analyze_failure(error, "parser", context)
        
        assert failure_event.failure_type == FailureType.CORRUPTION
    
    def test_analyze_dependency_failure(self):
        """Test classification of dependency failures."""
        error = Exception("External API service unavailable")
        context = {"api_endpoint": "https://external.api"}
        
        failure_event = self.analyzer.analyze_failure(error, "api_client", context)
        
        assert failure_event.failure_type == FailureType.DEPENDENCY
    
    def test_pattern_frequency_tracking(self):
        """Test that error patterns are tracked."""
        # Simulate multiple connection errors
        for i in range(3):
            error = ConnectionError(f"Connection error {i}")
            self.analyzer.analyze_failure(error, "network", {})
        
        # Check pattern frequency
        assert self.analyzer.pattern_frequency.get('connection_error', 0) == 3
    
    def test_get_recovery_recommendations_transient(self):
        """Test recovery recommendations for transient failures."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.TRANSIENT,
            component="test",
            error_message="Network timeout"
        )
        
        recommendations = self.analyzer.get_recovery_recommendations(failure_event)
        
        assert RecoveryAction.RETRY in recommendations
    
    def test_get_recovery_recommendations_resource(self):
        """Test recovery recommendations for resource failures."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.RESOURCE,
            component="test",
            error_message="Out of memory"
        )
        
        recommendations = self.analyzer.get_recovery_recommendations(failure_event)
        
        assert RecoveryAction.SCALE_RESOURCES in recommendations
        assert RecoveryAction.RESTART_COMPONENT in recommendations
    
    def test_get_recovery_recommendations_security(self):
        """Test recovery recommendations for security failures."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.SECURITY,
            component="test",
            error_message="Unauthorized access"
        )
        
        recommendations = self.analyzer.get_recovery_recommendations(failure_event)
        
        assert RecoveryAction.QUARANTINE in recommendations
        assert RecoveryAction.NOTIFY_ADMIN in recommendations
    
    def test_frequency_based_recommendations(self):
        """Test that high frequency patterns affect recommendations."""
        # Simulate many timeout errors
        self.analyzer.pattern_frequency['timeout_error'] = 10
        
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.TRANSIENT,
            component="test",
            error_message="Another timeout"
        )
        
        recommendations = self.analyzer.get_recovery_recommendations(failure_event)
        
        # Should recommend scaling due to high timeout frequency
        assert RecoveryAction.SCALE_RESOURCES in recommendations


class TestSelfHealingOrchestrator:
    """Test self-healing orchestration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.orchestrator = SelfHealingOrchestrator()
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.analyzer is not None
        assert len(self.orchestrator.action_handlers) > 0
        assert len(self.orchestrator.active_recoveries) == 0
    
    def test_register_component_restarter(self):
        """Test registering component restart handlers."""
        def test_restarter():
            return True
        
        self.orchestrator.register_component_restarter("test_component", test_restarter)
        
        assert "test_component" in self.orchestrator.component_restarters
        assert self.orchestrator.component_restarters["test_component"] == test_restarter
    
    def test_handle_failure_basic(self):
        """Test basic failure handling."""
        error = ConnectionError("Test connection error")
        
        failure_id = self.orchestrator.handle_failure(error, "test_component", {})
        
        assert failure_id.startswith("failure_")
        assert failure_id in self.orchestrator.active_recoveries
    
    def test_handle_failure_cooldown(self):
        """Test that cooldown periods are respected."""
        error = ConnectionError("Test error")
        component = "test_component"
        
        # First failure
        failure_id1 = self.orchestrator.handle_failure(error, component, {})
        
        # Immediate second failure should be in cooldown
        failure_id2 = self.orchestrator.handle_failure(error, component, {})
        
        # Should return the same failure ID (in cooldown)
        assert failure_id1 == failure_id2
    
    def test_handle_retry_action(self):
        """Test retry recovery action."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.TRANSIENT,
            component="test",
            error_message="Test error"
        )
        
        result = self.orchestrator._handle_retry(failure_event, RecoveryAction.RETRY)
        
        assert result is True
    
    def test_handle_restart_component_success(self):
        """Test successful component restart."""
        # Register a successful restarter
        def successful_restarter():
            return True
        
        self.orchestrator.register_component_restarter("test_component", successful_restarter)
        
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.RESOURCE,
            component="test_component",
            error_message="Component failed"
        )
        
        result = self.orchestrator._handle_restart_component(failure_event, RecoveryAction.RESTART_COMPONENT)
        
        assert result is True
    
    def test_handle_restart_component_failure(self):
        """Test failed component restart."""
        # Register a failing restarter
        def failing_restarter():
            return False
        
        self.orchestrator.register_component_restarter("test_component", failing_restarter)
        
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.RESOURCE,
            component="test_component",
            error_message="Component failed"
        )
        
        result = self.orchestrator._handle_restart_component(failure_event, RecoveryAction.RESTART_COMPONENT)
        
        assert result is False
    
    def test_handle_restart_component_no_restarter(self):
        """Test restart when no restarter is registered."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.RESOURCE,
            component="unknown_component",
            error_message="Component failed"
        )
        
        result = self.orchestrator._handle_restart_component(failure_event, RecoveryAction.RESTART_COMPONENT)
        
        assert result is False
    
    @patch('crewai_email_triage.advanced_error_recovery.get_smart_cache')
    def test_handle_reset_state(self, mock_get_cache):
        """Test state reset action."""
        mock_cache = Mock()
        mock_get_cache.return_value = mock_cache
        
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.CORRUPTION,
            component="test_component",
            error_message="State corrupted"
        )
        
        result = self.orchestrator._handle_reset_state(failure_event, RecoveryAction.RESET_STATE)
        
        assert result is True
        mock_cache.clear_all.assert_called_once()
    
    def test_handle_scale_resources(self):
        """Test resource scaling action."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.RESOURCE,
            component="test_component",
            error_message="Resource exhausted"
        )
        
        result = self.orchestrator._handle_scale_resources(failure_event, RecoveryAction.SCALE_RESOURCES)
        
        assert result is True
    
    def test_handle_fallback_mode(self):
        """Test fallback mode activation."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.DEPENDENCY,
            component="api_client",
            error_message="External service unavailable"
        )
        
        result = self.orchestrator._handle_fallback_mode(failure_event, RecoveryAction.FALLBACK_MODE)
        
        assert result is True
    
    def test_handle_quarantine(self):
        """Test quarantine action."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.SECURITY,
            component="suspicious_component",
            error_message="Security violation detected"
        )
        
        result = self.orchestrator._handle_quarantine(failure_event, RecoveryAction.QUARANTINE)
        
        assert result is True
    
    def test_handle_notify_admin(self):
        """Test admin notification action."""
        failure_event = FailureEvent(
            id="test",
            timestamp=datetime.utcnow(),
            failure_type=FailureType.CONFIGURATION,
            component="critical_service",
            error_message="Critical configuration error"
        )
        
        result = self.orchestrator._handle_notify_admin(failure_event, RecoveryAction.NOTIFY_ADMIN)
        
        assert result is True
    
    def test_get_recovery_status_active(self):
        """Test getting recovery status for active recovery."""
        # Mock an active recovery
        mock_future = Mock()
        mock_future.done.return_value = False
        
        failure_id = "test_failure_123"
        self.orchestrator.active_recoveries[failure_id] = mock_future
        
        status = self.orchestrator.get_recovery_status(failure_id)
        
        assert status is not None
        assert status["failure_id"] == failure_id
        assert status["status"] == "in_progress"
    
    def test_get_recovery_status_completed(self):
        """Test getting recovery status for completed recovery."""
        # Mock a completed recovery
        mock_future = Mock()
        mock_future.done.return_value = True
        mock_future.result.return_value = True
        
        failure_id = "test_failure_123"
        self.orchestrator.active_recoveries[failure_id] = mock_future
        
        status = self.orchestrator.get_recovery_status(failure_id)
        
        assert status is not None
        assert status["status"] == "completed"
        assert status["result"] is True
    
    def test_get_recovery_status_not_found(self):
        """Test getting recovery status for non-existent failure."""
        status = self.orchestrator.get_recovery_status("non_existent_failure")
        
        assert status is None
    
    def test_get_system_health_report(self):
        """Test system health report generation."""
        # Add some failure history
        recent_failure = FailureEvent(
            id="recent",
            timestamp=datetime.utcnow() - timedelta(minutes=30),
            failure_type=FailureType.TRANSIENT,
            component="test",
            error_message="Recent error",
            resolved=True,
            resolution_time=datetime.utcnow() - timedelta(minutes=25)
        )
        
        old_failure = FailureEvent(
            id="old",
            timestamp=datetime.utcnow() - timedelta(hours=2),
            failure_type=FailureType.RESOURCE,
            component="test",
            error_message="Old error"
        )
        
        self.orchestrator.analyzer.failure_history = [recent_failure, old_failure]
        
        report = self.orchestrator.get_system_health_report()
        
        assert "total_failures_last_hour" in report
        assert "resolved_failures" in report
        assert "resolution_rate" in report
        assert "failure_types" in report
        assert "active_recoveries" in report
        
        # Should only count recent failure
        assert report["total_failures_last_hour"] == 1
        assert report["resolved_failures"] == 1
        assert report["resolution_rate"] == 1.0


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_self_healing_orchestrator_singleton(self):
        """Test that orchestrator is singleton."""
        orchestrator1 = get_self_healing_orchestrator()
        orchestrator2 = get_self_healing_orchestrator()
        
        assert orchestrator1 is orchestrator2
    
    @patch('crewai_email_triage.advanced_error_recovery.get_self_healing_orchestrator')
    def test_handle_failure_with_recovery(self, mock_get_orchestrator):
        """Test convenience function for failure handling."""
        mock_orchestrator = Mock()
        mock_orchestrator.handle_failure.return_value = "failure_123"
        mock_get_orchestrator.return_value = mock_orchestrator
        
        error = Exception("Test error")
        failure_id = handle_failure_with_recovery(error, "test_component", {"key": "value"})
        
        assert failure_id == "failure_123"
        mock_orchestrator.handle_failure.assert_called_once_with(
            error, "test_component", {"key": "value"}
        )


class TestIntegration:
    """Integration tests for error recovery system."""
    
    def test_full_error_recovery_workflow(self):
        """Test complete error recovery workflow."""
        orchestrator = SelfHealingOrchestrator()
        
        # Register a component restarter
        restart_called = False
        
        def test_restarter():
            nonlocal restart_called
            restart_called = True
            return True
        
        orchestrator.register_component_restarter("test_service", test_restarter)
        
        # Simulate a resource failure
        error = MemoryError("Out of memory in test_service")
        failure_id = orchestrator.handle_failure(error, "test_service", {"memory_usage": "high"})
        
        # Wait for recovery to complete
        time.sleep(0.1)
        
        # Check that recovery was attempted
        assert failure_id in orchestrator.active_recoveries
        
        # Get the failure from history
        failures = [f for f in orchestrator.analyzer.failure_history if f.id == failure_id]
        assert len(failures) == 1
        
        failure = failures[0]
        assert failure.failure_type == FailureType.RESOURCE
        assert failure.component == "test_service"
    
    def test_multiple_failures_same_component(self):
        """Test handling multiple failures from same component."""
        orchestrator = SelfHealingOrchestrator()
        
        error1 = ConnectionError("First connection error")
        error2 = ConnectionError("Second connection error")
        
        failure_id1 = orchestrator.handle_failure(error1, "network_client", {})
        failure_id2 = orchestrator.handle_failure(error2, "network_client", {})
        
        # Second failure should be in cooldown (same ID)
        assert failure_id1 == failure_id2
    
    def test_different_failure_types_different_recovery(self):
        """Test that different failure types get different recovery strategies."""
        orchestrator = SelfHealingOrchestrator()
        
        # Security failure
        security_error = PermissionError("Unauthorized access")
        security_failure = orchestrator.analyzer.analyze_failure(
            security_error, "auth_service", {}
        )
        security_recs = orchestrator.analyzer.get_recovery_recommendations(security_failure)
        
        # Resource failure
        resource_error = MemoryError("Out of memory")
        resource_failure = orchestrator.analyzer.analyze_failure(
            resource_error, "processor", {}
        )
        resource_recs = orchestrator.analyzer.get_recovery_recommendations(resource_failure)
        
        # Should have different recovery strategies
        assert RecoveryAction.QUARANTINE in security_recs
        assert RecoveryAction.SCALE_RESOURCES in resource_recs
        assert RecoveryAction.QUARANTINE not in resource_recs


@pytest.mark.performance
class TestPerformanceErrorRecovery:
    """Performance tests for error recovery system."""
    
    def test_high_frequency_error_handling(self):
        """Test handling high frequency of errors."""
        orchestrator = SelfHealingOrchestrator()
        
        start_time = time.time()
        
        # Generate many errors quickly
        for i in range(100):
            error = Exception(f"Error {i}")
            orchestrator.handle_failure(error, f"component_{i % 10}", {})
        
        handling_time = time.time() - start_time
        
        # Should handle 100 errors quickly
        assert handling_time < 1.0
        assert len(orchestrator.analyzer.failure_history) == 100
    
    def test_memory_usage_with_many_failures(self):
        """Test memory usage doesn't grow excessively with many failures."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        analyzer = ErrorPatternAnalyzer()
        
        # Add many failures
        for i in range(1000):
            error = Exception(f"Test error {i}")
            analyzer.analyze_failure(error, "test_component", {})
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / initial_memory
        
        # Memory growth should be reasonable
        assert memory_growth < 0.3  # Less than 30% growth


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
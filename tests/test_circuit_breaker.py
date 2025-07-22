#!/usr/bin/env python3
"""
Test suite for circuit breaker pattern implementation.

Tests the CircuitBreaker class to ensure:
- Proper state transitions (Closed -> Open -> Half-Open -> Closed)
- Failure threshold detection and fast-fail behavior
- Recovery mechanisms and timeout handling
- Integration with retry logic
- Thread safety for concurrent operations
"""

import os
import sys
import unittest
import time
from unittest.mock import patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
import threading

# Add project root to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crewai_email_triage.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError


class TestCircuitBreakerConfig(unittest.TestCase):
    """Test circuit breaker configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        self.assertEqual(config.failure_threshold, 5)
        self.assertEqual(config.recovery_timeout, 60.0)
        self.assertEqual(config.success_threshold, 3)
        self.assertIsInstance(config.monitored_exceptions, tuple)
        self.assertIn(ConnectionError, config.monitored_exceptions)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120.0,
            success_threshold=5,
            monitored_exceptions=(ValueError, OSError)
        )
        self.assertEqual(config.failure_threshold, 10)
        self.assertEqual(config.recovery_timeout, 120.0)
        self.assertEqual(config.success_threshold, 5)
        self.assertEqual(config.monitored_exceptions, (ValueError, OSError))
    
    def test_config_validation(self):
        """Test configuration validation."""
        with self.assertRaises(ValueError):
            CircuitBreakerConfig(failure_threshold=0)
        
        with self.assertRaises(ValueError):
            CircuitBreakerConfig(recovery_timeout=-1)
        
        with self.assertRaises(ValueError):
            CircuitBreakerConfig(success_threshold=0)
    
    def test_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            'CIRCUIT_BREAKER_FAILURE_THRESHOLD': '8',
            'CIRCUIT_BREAKER_RECOVERY_TIMEOUT': '90.0',
            'CIRCUIT_BREAKER_SUCCESS_THRESHOLD': '4'
        }):
            config = CircuitBreakerConfig.from_env()
            self.assertEqual(config.failure_threshold, 8)
            self.assertEqual(config.recovery_timeout, 90.0)
            self.assertEqual(config.success_threshold, 4)


class TestCircuitBreaker(unittest.TestCase):
    """Test circuit breaker behavior."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            monitored_exceptions=(ConnectionError, TimeoutError)
        )
        self.circuit_breaker = CircuitBreaker("test_service", self.config)
    
    def test_initial_state_closed(self):
        """Test that circuit breaker starts in closed state."""
        self.assertEqual(self.circuit_breaker.state, "closed")
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertEqual(self.circuit_breaker.success_count, 0)
    
    def test_successful_operation_in_closed_state(self):
        """Test successful operation when circuit is closed."""
        def successful_operation():
            return "success"
        
        result = self.circuit_breaker.call(successful_operation)
        self.assertEqual(result, "success")
        self.assertEqual(self.circuit_breaker.state, "closed")
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_failed_operation_increments_failure_count(self):
        """Test that failed operations increment failure count."""
        def failing_operation():
            raise ConnectionError("Network error")
        
        with self.assertRaises(ConnectionError):
            self.circuit_breaker.call(failing_operation)
        
        self.assertEqual(self.circuit_breaker.failure_count, 1)
        self.assertEqual(self.circuit_breaker.state, "closed")
    
    def test_circuit_opens_after_failure_threshold(self):
        """Test that circuit opens after reaching failure threshold."""
        def failing_operation():
            raise ConnectionError("Network error")
        
        # Execute failures up to threshold
        for i in range(self.config.failure_threshold):
            with self.assertRaises(ConnectionError):
                self.circuit_breaker.call(failing_operation)
        
        self.assertEqual(self.circuit_breaker.state, "open")
        self.assertEqual(self.circuit_breaker.failure_count, self.config.failure_threshold)
    
    def test_fast_fail_when_circuit_open(self):
        """Test fast-fail behavior when circuit is open."""
        def failing_operation():
            raise ConnectionError("Network error")
        
        # Open the circuit
        for i in range(self.config.failure_threshold):
            with self.assertRaises(ConnectionError):
                self.circuit_breaker.call(failing_operation)
        
        # Now it should fast-fail
        with self.assertRaises(CircuitBreakerError):
            self.circuit_breaker.call(failing_operation)
    
    def test_circuit_transitions_to_half_open(self):
        """Test transition from open to half-open after timeout."""
        def failing_operation():
            raise ConnectionError("Network error")
        
        # Open the circuit
        for i in range(self.config.failure_threshold):
            with self.assertRaises(ConnectionError):
                self.circuit_breaker.call(failing_operation)
        
        self.assertEqual(self.circuit_breaker.state, "open")
        
        # Wait for recovery timeout
        time.sleep(self.config.recovery_timeout + 0.1)
        
        # Next call should transition to half-open
        with self.assertRaises(ConnectionError):
            self.circuit_breaker.call(failing_operation)
        
        self.assertEqual(self.circuit_breaker.state, "open")  # Failed, so back to open
    
    def test_circuit_closes_after_successful_half_open(self):
        """Test circuit closes after successful operations in half-open state."""
        def failing_operation():
            raise ConnectionError("Network error")
        
        def successful_operation():
            return "success"
        
        # Open the circuit
        for i in range(self.config.failure_threshold):
            with self.assertRaises(ConnectionError):
                self.circuit_breaker.call(failing_operation)
        
        # Wait for recovery timeout
        time.sleep(self.config.recovery_timeout + 0.1)
        
        # Successful operations in half-open state
        for i in range(self.config.success_threshold):
            result = self.circuit_breaker.call(successful_operation)
            self.assertEqual(result, "success")
        
        self.assertEqual(self.circuit_breaker.state, "closed")
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_non_monitored_exceptions_pass_through(self):
        """Test that non-monitored exceptions don't affect circuit state."""
        def operation_with_non_monitored_exception():
            raise ValueError("Not a monitored exception")
        
        with self.assertRaises(ValueError):
            self.circuit_breaker.call(operation_with_non_monitored_exception)
        
        # Circuit should remain closed and failure count unchanged
        self.assertEqual(self.circuit_breaker.state, "closed")
        self.assertEqual(self.circuit_breaker.failure_count, 0)
    
    def test_thread_safety(self):
        """Test circuit breaker thread safety."""
        results = []
        errors = []
        
        def thread_operation(thread_id):
            try:
                if thread_id % 2 == 0:
                    # Even threads succeed
                    result = self.circuit_breaker.call(lambda: f"success_{thread_id}")
                    results.append(result)
                else:
                    # Odd threads fail
                    self.circuit_breaker.call(lambda: (_ for _ in ()).throw(ConnectionError("fail")))
            except Exception as e:
                errors.append((thread_id, type(e).__name__))
        
        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(thread_operation, i) for i in range(20)]
            for future in futures:
                future.result()
        
        # Verify that operations completed and state is consistent
        self.assertGreater(len(results), 0)
        self.assertGreater(len(errors), 0)
        # State should be deterministic based on the pattern of successes/failures
        self.assertIn(self.circuit_breaker.state, ["closed", "open", "half_open"])
    
    def test_reset_circuit_breaker(self):
        """Test manual reset of circuit breaker."""
        def failing_operation():
            raise ConnectionError("Network error")
        
        # Open the circuit
        for i in range(self.config.failure_threshold):
            with self.assertRaises(ConnectionError):
                self.circuit_breaker.call(failing_operation)
        
        self.assertEqual(self.circuit_breaker.state, "open")
        
        # Reset the circuit
        self.circuit_breaker.reset()
        
        self.assertEqual(self.circuit_breaker.state, "closed")
        self.assertEqual(self.circuit_breaker.failure_count, 0)
        self.assertEqual(self.circuit_breaker.success_count, 0)
    
    def test_circuit_breaker_metrics(self):
        """Test that circuit breaker maintains proper metrics."""
        def successful_operation():
            return "success"
        
        def failing_operation():
            raise ConnectionError("Network error")
        
        # Execute some operations
        self.circuit_breaker.call(successful_operation)
        
        try:
            self.circuit_breaker.call(failing_operation)
        except ConnectionError:
            pass
        
        # Check metrics after failure
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["state"], "closed")
        self.assertEqual(metrics["failure_count"], 1)
        self.assertEqual(metrics["total_calls"], 2)
        self.assertIsNotNone(metrics["last_failure_time"])
        
        # Successful operation resets failure count in closed state
        self.circuit_breaker.call(successful_operation)
        
        # Check metrics after success - failure count should be reset
        metrics = self.circuit_breaker.get_metrics()
        self.assertEqual(metrics["state"], "closed")
        self.assertEqual(metrics["failure_count"], 0)  # Reset by success
        self.assertEqual(metrics["total_calls"], 3)
        self.assertIsNotNone(metrics["last_failure_time"])  # Still recorded


class TestCircuitBreakerIntegration(unittest.TestCase):
    """Test circuit breaker integration with other systems."""
    
    def test_circuit_breaker_with_retry_logic(self):
        """Test circuit breaker working with retry logic."""
        from crewai_email_triage.retry_utils import retry_with_backoff, RetryConfig, CircuitBreakerError
        from crewai_email_triage.circuit_breaker import CircuitBreakerConfig
        
        # Configure retry with circuit breaker
        circuit_config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.5)
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.1,
            enable_circuit_breaker=True,
            circuit_breaker_config=circuit_config
        )
        
        call_count = 0
        
        @retry_with_backoff(retry_config, "test_integration")
        def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Network failure")
        
        # First set of failures should trigger retries and eventually open circuit
        with self.assertRaises(ConnectionError):
            failing_operation()
        
        # Reset call count
        call_count = 0
        
        # Second failure should open the circuit
        with self.assertRaises(ConnectionError):
            failing_operation()
        
        # Reset call count
        call_count = 0
        
        # Now circuit should be open and should fast-fail without retries
        with self.assertRaises(CircuitBreakerError):
            failing_operation()
        
        # Should not have called the function because circuit is open
        self.assertEqual(call_count, 0)
    
    def test_circuit_breaker_recovery_with_retry(self):
        """Test circuit breaker recovery working with retry logic."""
        from crewai_email_triage.retry_utils import retry_with_backoff, RetryConfig
        from crewai_email_triage.circuit_breaker import CircuitBreakerConfig
        
        # Configure with short timeouts for testing
        circuit_config = CircuitBreakerConfig(
            failure_threshold=2, 
            recovery_timeout=0.2,
            success_threshold=1
        )
        retry_config = RetryConfig(
            max_attempts=2,
            base_delay=0.05,
            enable_circuit_breaker=True,
            circuit_breaker_config=circuit_config
        )
        
        call_count = 0
        should_fail = True
        
        @retry_with_backoff(retry_config, "test_recovery")
        def operation():
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ConnectionError("Network failure")
            return "success"
        
        # Open the circuit with failures
        with self.assertRaises(ConnectionError):
            operation()
        call_count = 0
        
        with self.assertRaises(ConnectionError):
            operation()
        call_count = 0
        
        # Wait for recovery timeout
        time.sleep(0.3)
        
        # Make operation succeed
        should_fail = False
        
        # Should recover and succeed
        result = operation()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)  # Should only call once, not retry
    
    def test_circuit_breaker_disabled(self):
        """Test retry logic works normally when circuit breaker is disabled."""
        from crewai_email_triage.retry_utils import retry_with_backoff, RetryConfig
        
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=0.05,
            enable_circuit_breaker=False  # Disabled
        )
        
        call_count = 0
        
        @retry_with_backoff(retry_config)
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network failure")
            return "success"
        
        # Should retry and eventually succeed
        result = failing_operation()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)  # Should have retried


if __name__ == "__main__":
    unittest.main()
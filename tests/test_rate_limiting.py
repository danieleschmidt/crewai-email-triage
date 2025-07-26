"""Comprehensive test suite for rate limiting and backpressure mechanisms."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, Mock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from crewai_email_triage.rate_limiter import (
    RateLimiter, RateLimitConfig, BatchRateLimiter,
    get_rate_limiter, reset_rate_limiter
)
from crewai_email_triage.env_config import reset_config_cache
from crewai_email_triage.pipeline import triage_email, triage_batch


class TestRateLimitConfig:
    """Test rate limit configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RateLimitConfig()
        
        assert config.requests_per_second == 10.0
        assert config.burst_size == 20
        assert config.enabled is True
        assert config.backpressure_threshold == 0.8
        assert config.backpressure_delay == 0.1
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RateLimitConfig(
            requests_per_second=5.0,
            burst_size=10,
            enabled=False,
            backpressure_threshold=0.5,
            backpressure_delay=0.2
        )
        
        assert config.requests_per_second == 5.0
        assert config.burst_size == 10
        assert config.enabled is False
        assert config.backpressure_threshold == 0.5
        assert config.backpressure_delay == 0.2


class TestRateLimiter:
    """Test rate limiter core functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        config = RateLimitConfig(requests_per_second=5.0, burst_size=10)
        limiter = RateLimiter(config)
        
        assert limiter._config.requests_per_second == 5.0
        assert limiter._config.burst_size == 10
        assert limiter._tokens == 10.0  # Start with full bucket
    
    def test_token_acquisition(self):
        """Test basic token acquisition."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)
        
        # Should be able to acquire tokens immediately from full bucket
        delay = limiter.acquire(1.0)
        assert delay == 0.0
        assert limiter._tokens == 4.0
    
    def test_token_acquisition_when_depleted(self):
        """Test token acquisition when bucket is depleted."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=2)
        limiter = RateLimiter(config)
        
        # Deplete the bucket
        limiter.acquire(2.0)
        assert limiter._tokens == 0.0
        
        # Should wait for token refill
        start = time.time()
        delay = limiter.acquire(1.0)
        elapsed = time.time() - start
        
        assert delay > 0
        assert elapsed >= delay * 0.9  # Allow small timing variance
        assert limiter._tokens == 0.0
    
    def test_try_acquire_success(self):
        """Test try_acquire when tokens are available."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)
        
        result = limiter.try_acquire(1.0)
        assert result is True
        assert limiter._tokens == 4.0
    
    def test_try_acquire_failure(self):
        """Test try_acquire when not enough tokens available."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=2)
        limiter = RateLimiter(config)
        
        # Deplete tokens
        limiter.acquire(2.0)
        
        # Should fail to acquire more tokens
        result = limiter.try_acquire(1.0)
        assert result is False
        # Tokens should be approximately 0 (allowing for tiny time differences)
        assert limiter._tokens < 0.1
    
    def test_disabled_rate_limiter(self):
        """Test rate limiter when disabled."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)
        
        # Should always succeed immediately when disabled
        delay = limiter.acquire(100.0)  # Large request
        assert delay == 0.0
        
        result = limiter.try_acquire(100.0)
        assert result is True
    
    def test_backpressure_activation(self):
        """Test backpressure activation when token ratio is low."""
        config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=10,
            backpressure_threshold=0.5,
            backpressure_delay=0.05
        )
        limiter = RateLimiter(config)
        
        # Use enough tokens to trigger backpressure (below 50% = 5 tokens)
        limiter.acquire(6.0)  # 4 tokens left, below threshold
        
        # Next acquisition should include backpressure delay
        start = time.time()
        delay = limiter.acquire(1.0)
        elapsed = time.time() - start
        
        assert delay >= config.backpressure_delay
        assert elapsed >= config.backpressure_delay * 0.9
    
    def test_token_refill(self):
        """Test token bucket refills over time."""
        config = RateLimitConfig(requests_per_second=20.0, burst_size=5)
        limiter = RateLimiter(config)
        
        # Deplete tokens
        limiter.acquire(5.0)
        assert limiter._tokens == 0.0
        
        # Wait for refill
        time.sleep(0.1)  # 0.1 seconds should add 2 tokens at 20/sec
        
        # Force refill check
        limiter._refill_tokens()
        assert limiter._tokens >= 1.8  # Allow for timing variance
    
    def test_get_status(self):
        """Test rate limiter status reporting."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=20)
        limiter = RateLimiter(config)
        
        status = limiter.get_status()
        
        assert status["enabled"] is True
        assert status["tokens_available"] == 20.0
        assert status["max_tokens"] == 20
        assert status["utilization"] == 0.0
        assert status["backpressure_active"] is False
        assert status["requests_per_second"] == 10.0
        
        # Use some tokens and check status
        limiter.acquire(5.0)
        status = limiter.get_status()
        
        # Allow for small timing variations
        assert abs(status["tokens_available"] - 15.0) < 0.1
        assert abs(status["utilization"] - 0.25) < 0.01  # 5/20 = 0.25
    
    def test_context_manager(self):
        """Test rate limiter context manager."""
        config = RateLimitConfig(requests_per_second=10.0, burst_size=5)
        limiter = RateLimiter(config)
        
        with limiter.rate_limited_operation(1.0) as delay:
            assert delay == 0.0  # First operation should be immediate
            assert limiter._tokens == 4.0


class TestThreadSafety:
    """Test rate limiter thread safety under concurrent access."""
    
    def test_concurrent_token_acquisition(self):
        """Test concurrent token acquisition from multiple threads."""
        config = RateLimitConfig(requests_per_second=50.0, burst_size=100)
        limiter = RateLimiter(config)
        
        def acquire_tokens():
            delays = []
            for _ in range(10):
                delay = limiter.acquire(1.0)
                delays.append(delay)
            return delays
        
        # Run multiple threads concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(acquire_tokens) for _ in range(5)]
            all_delays = []
            for future in as_completed(futures):
                all_delays.extend(future.result())
        
        # Total tokens acquired: 5 threads * 10 tokens = 50 tokens
        # With refilling, final token count may vary, but should be reasonable
        status = limiter.get_status()
        assert status["tokens_available"] <= 100  # Can't exceed bucket size
        
        # Most acquisitions should be immediate due to large bucket
        immediate_acquisitions = sum(1 for delay in all_delays if delay == 0.0)
        assert immediate_acquisitions >= 40  # Most should be immediate
    
    def test_thread_safety_under_heavy_load(self):
        """Test thread safety under heavy concurrent load."""
        config = RateLimitConfig(requests_per_second=100.0, burst_size=20)
        limiter = RateLimiter(config)
        
        results = []
        
        def heavy_load_worker():
            worker_results = []
            for _ in range(50):
                success = limiter.try_acquire(1.0)
                worker_results.append(success)
            return worker_results
        
        # Run many threads to stress test
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(heavy_load_worker) for _ in range(10)]
            for future in as_completed(futures):
                results.extend(future.result())
        
        # Should have some successes and some failures due to rate limiting
        successes = sum(1 for result in results if result)
        failures = sum(1 for result in results if not result)
        
        assert successes > 0
        assert failures > 0
        assert successes + failures == 500  # 10 threads * 50 attempts
    
    def test_concurrent_status_access(self):
        """Test concurrent access to status information."""
        config = RateLimitConfig(requests_per_second=20.0, burst_size=40)
        limiter = RateLimiter(config)
        
        def status_worker():
            statuses = []
            for _ in range(20):
                limiter.acquire(0.5)
                status = limiter.get_status()
                statuses.append(status)
                time.sleep(0.01)
            return statuses
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(status_worker) for _ in range(3)]
            all_statuses = []
            for future in as_completed(futures):
                all_statuses.extend(future.result())
        
        # All status calls should complete successfully
        assert len(all_statuses) == 60  # 3 threads * 20 statuses
        for status in all_statuses:
            assert "enabled" in status
            assert "tokens_available" in status
            assert status["tokens_available"] >= 0


class TestBatchRateLimiter:
    """Test batch rate limiter functionality."""
    
    def test_batch_processing_disabled(self):
        """Test batch processing when rate limiting is disabled."""
        config = RateLimitConfig(enabled=False)
        batch_limiter = BatchRateLimiter(config)
        
        items = ["item1", "item2", "item3"]
        results = []
        
        def process_item(item):
            return item.upper()
        
        for result in batch_limiter.process_batch_with_rate_limiting(
            len(items), process_item, items
        ):
            results.append(result)
        
        assert results == ["ITEM1", "ITEM2", "ITEM3"]
    
    def test_batch_processing_with_rate_limiting(self):
        """Test batch processing with rate limiting enabled."""
        config = RateLimitConfig(
            requests_per_second=10.0,
            burst_size=3,
            enabled=True
        )
        batch_limiter = BatchRateLimiter(config)
        
        items = ["item1", "item2", "item3", "item4"]
        results = []
        
        def process_item(item):
            return item.upper()
        
        start = time.time()
        for result in batch_limiter.process_batch_with_rate_limiting(
            len(items), process_item, items
        ):
            results.append(result)
        elapsed = time.time() - start
        
        assert results == ["ITEM1", "ITEM2", "ITEM3", "ITEM4"]
        # Should take some time due to rate limiting after burst is exhausted
        assert elapsed > 0.05  # Some delay expected


class TestPipelineIntegration:
    """Test rate limiting integration with pipeline functions."""
    
    def test_triage_email_with_rate_limiting(self):
        """Test triage_email function with rate limiting enabled."""
        with patch.dict(os.environ, {
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_REQUESTS_PER_SECOND": "20.0",
            "RATE_LIMIT_BURST_SIZE": "5"
        }):
            # Reset config cache to pick up new environment
            reset_config_cache()
            reset_rate_limiter()
            
            email_content = "Test email for rate limiting"
            
            start = time.time()
            result = triage_email(email_content, enable_rate_limiting=True)
            elapsed = time.time() - start
            
            # Should process successfully
            assert "category" in result
            assert "priority" in result
            
            # Should be relatively fast for first request
            assert elapsed < 1.0
    
    def test_triage_email_rate_limiting_disabled(self):
        """Test triage_email function with rate limiting explicitly disabled."""
        email_content = "Test email without rate limiting"
        
        result = triage_email(email_content, enable_rate_limiting=False)
        
        # Should process successfully
        assert "category" in result
        assert "priority" in result
    
    def test_triage_batch_with_rate_limiting_sequential(self):
        """Test triage_batch function with rate limiting in sequential mode."""
        with patch.dict(os.environ, {
            "RATE_LIMIT_ENABLED": "true",
            "RATE_LIMIT_REQUESTS_PER_SECOND": "50.0",
            "RATE_LIMIT_BURST_SIZE": "10"
        }):
            reset_config_cache()
            reset_rate_limiter()
            
            messages = [
                "First test email",
                "Second test email", 
                "Third test email"
            ]
            
            results = triage_batch(
                messages, 
                parallel=False, 
                enable_rate_limiting=True
            )
            
            assert len(results) == 3
            for result in results:
                assert "category" in result
                assert "priority" in result
    
    def test_triage_batch_with_rate_limiting_parallel(self):
        """Test triage_batch function with rate limiting in parallel mode."""
        with patch.dict(os.environ, {
            "RATE_LIMIT_ENABLED": "true", 
            "RATE_LIMIT_REQUESTS_PER_SECOND": "30.0",
            "RATE_LIMIT_BURST_SIZE": "8"
        }):
            reset_config_cache()
            reset_rate_limiter()
            
            messages = [
                "Parallel test email 1",
                "Parallel test email 2",
                "Parallel test email 3",
                "Parallel test email 4"
            ]
            
            results = triage_batch(
                messages,
                parallel=True,
                max_workers=2,
                enable_rate_limiting=True
            )
            
            assert len(results) == 4
            for result in results:
                assert "category" in result
                assert "priority" in result
    
    def test_pipeline_rate_limiting_environment_default(self):
        """Test pipeline functions use environment configuration by default."""
        with patch.dict(os.environ, {
            "RATE_LIMIT_ENABLED": "false"
        }):
            reset_config_cache()
            reset_rate_limiter()
            
            # Should use environment default (disabled)
            result = triage_email("Test email")
            assert "category" in result
            
            # Should process quickly when disabled
            messages = ["Email 1", "Email 2"]
            start = time.time()
            results = triage_batch(messages)
            elapsed = time.time() - start
            
            assert len(results) == 2
            assert elapsed < 0.5  # Should be fast when rate limiting disabled


class TestEnvironmentConfiguration:
    """Test rate limiting environment configuration."""
    
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            "RATE_LIMIT_REQUESTS_PER_SECOND": "15.0",
            "RATE_LIMIT_BURST_SIZE": "30",
            "RATE_LIMIT_ENABLED": "false",
            "RATE_LIMIT_BACKPRESSURE_THRESHOLD": "0.6",
            "RATE_LIMIT_BACKPRESSURE_DELAY": "0.2"
        }):
            reset_config_cache()
            reset_rate_limiter()
            
            limiter = get_rate_limiter()
            config = limiter._config
            
            assert config.requests_per_second == 15.0
            assert config.burst_size == 30
            assert config.enabled is False
            assert config.backpressure_threshold == 0.6
            assert config.backpressure_delay == 0.2
    
    def test_invalid_environment_values_fallback(self):
        """Test fallback behavior with invalid environment values."""
        with patch.dict(os.environ, {
            "RATE_LIMIT_REQUESTS_PER_SECOND": "invalid",
            "RATE_LIMIT_BURST_SIZE": "not_a_number"
        }):
            # Should fall back to defaults when invalid values provided
            reset_config_cache()
            
            try:
                reset_rate_limiter()
                limiter = get_rate_limiter()
                # Should either use defaults or raise appropriate error
                assert limiter is not None
            except (ValueError, TypeError):
                # Expected behavior for invalid config
                pass


class TestMetricsIntegration:
    """Test rate limiting metrics integration."""
    
    def test_rate_limiter_metrics_collection(self):
        """Test that rate limiting events generate appropriate metrics."""
        with patch("crewai_email_triage.pipeline._metrics_collector") as mock_metrics:
            mock_metrics.increment_counter = Mock()
            mock_metrics.record_histogram = Mock()
            mock_metrics.set_gauge = Mock()
            
            email_content = "Test email for metrics"
            triage_email(email_content, enable_rate_limiting=True)
            
            # Verify metrics were called
            assert mock_metrics.increment_counter.called
            assert mock_metrics.set_gauge.called
    
    def test_rate_limit_delay_metrics(self):
        """Test metrics collection for rate limit delays."""
        config = RateLimitConfig(requests_per_second=1.0, burst_size=1)  # Very restrictive
        limiter = RateLimiter(config)
        
        with patch("crewai_email_triage.pipeline._metrics_collector") as mock_metrics:
            mock_metrics.increment_counter = Mock()
            mock_metrics.record_histogram = Mock()
            
            # Deplete tokens and force delay
            limiter.acquire(1.0)
            
            # This should cause a delay and generate metrics
            with limiter.rate_limited_operation(1.0):
                pass


if __name__ == "__main__":
    # Run basic smoke tests
    test_config = TestRateLimitConfig()
    test_config.test_default_config()
    test_config.test_custom_config()
    
    test_limiter = TestRateLimiter()
    test_limiter.test_rate_limiter_initialization()
    test_limiter.test_token_acquisition()
    test_limiter.test_disabled_rate_limiter()
    
    test_thread_safety = TestThreadSafety()
    test_thread_safety.test_concurrent_token_acquisition()
    
    print("✅ All basic rate limiting tests passed!")
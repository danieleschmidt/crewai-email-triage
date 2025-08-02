"""Performance testing configuration and benchmarks."""

import pytest
import time
from typing import Dict, Any, List
from unittest.mock import patch, Mock


class PerformanceBenchmarks:
    """Performance benchmark thresholds for different operations."""
    
    # Single email processing thresholds (seconds)
    SINGLE_EMAIL_CLASSIFICATION = 0.5
    SINGLE_EMAIL_PRIORITIZATION = 0.3
    SINGLE_EMAIL_SUMMARIZATION = 1.0
    SINGLE_EMAIL_RESPONSE = 0.8
    SINGLE_EMAIL_COMPLETE_PIPELINE = 2.0
    
    # Batch processing thresholds (emails per second)
    BATCH_PROCESSING_MIN_THROUGHPUT = 10
    PARALLEL_BATCH_MIN_THROUGHPUT = 25
    
    # Memory usage thresholds (MB)
    SINGLE_EMAIL_MAX_MEMORY = 50
    BATCH_PROCESSING_MAX_MEMORY = 200
    
    # System resource thresholds
    MAX_CPU_USAGE_PERCENT = 80
    MAX_MEMORY_USAGE_PERCENT = 70


@pytest.fixture
def performance_thresholds():
    """Performance threshold configuration."""
    return PerformanceBenchmarks()


@pytest.fixture
def performance_test_data():
    """Generate test data for performance testing."""
    def generate_emails(count: int) -> List[Dict[str, Any]]:
        emails = []
        for i in range(count):
            emails.append({
                "id": f"perf_test_{i:06d}",
                "subject": f"Performance test email {i}",
                "sender": f"sender{i % 10}@test.com",
                "recipient": "recipient@test.com",
                "body": f"This is test email number {i} for performance testing. " * 10,
                "timestamp": f"2025-08-02T{(10 + i % 14):02d}:00:00Z",
                "headers": {"Message-ID": f"perf-{i}@test.com"},
                "attachments": []
            })
        return emails
    
    return {
        "small_batch": generate_emails(10),
        "medium_batch": generate_emails(100),
        "large_batch": generate_emails(1000),
        "xlarge_batch": generate_emails(5000)
    }


@pytest.fixture
def memory_profiler():
    """Memory profiling utilities."""
    try:
        import psutil
        import os
        
        class MemoryProfiler:
            def __init__(self):
                self.process = psutil.Process(os.getpid())
                self.baseline_memory = self.get_memory_usage()
                
            def get_memory_usage(self) -> float:
                """Get current memory usage in MB."""
                return self.process.memory_info().rss / 1024 / 1024
                
            def get_cpu_usage(self) -> float:
                """Get current CPU usage percentage."""
                return self.process.cpu_percent()
                
            def memory_delta(self) -> float:
                """Get memory usage delta from baseline in MB."""
                return self.get_memory_usage() - self.baseline_memory
                
            def reset_baseline(self):
                """Reset memory baseline."""
                self.baseline_memory = self.get_memory_usage()
        
        return MemoryProfiler()
        
    except ImportError:
        # Fallback mock profiler if psutil not available
        class MockMemoryProfiler:
            def get_memory_usage(self): return 0.0
            def get_cpu_usage(self): return 0.0
            def memory_delta(self): return 0.0
            def reset_baseline(self): pass
            
        return MockMemoryProfiler()


@pytest.fixture
def performance_monitor():
    """Enhanced performance monitoring with detailed metrics."""
    class PerformanceMonitor:
        def __init__(self):
            self.timers = {}
            self.counters = {}
            self.memory_snapshots = {}
            
        def start_timer(self, operation: str):
            """Start timing an operation."""
            self.timers[operation] = {"start": time.perf_counter()}
            
        def end_timer(self, operation: str) -> float:
            """End timing an operation and return duration."""
            if operation not in self.timers:
                raise ValueError(f"Timer '{operation}' was not started")
                
            end_time = time.perf_counter()
            duration = end_time - self.timers[operation]["start"]
            self.timers[operation]["duration"] = duration
            return duration
            
        def get_duration(self, operation: str) -> float:
            """Get duration of a completed operation."""
            return self.timers.get(operation, {}).get("duration", 0.0)
            
        def increment_counter(self, name: str, value: int = 1):
            """Increment a counter."""
            self.counters[name] = self.counters.get(name, 0) + value
            
        def get_counter(self, name: str) -> int:
            """Get counter value."""
            return self.counters.get(name, 0)
            
        def take_memory_snapshot(self, name: str, memory_profiler):
            """Take a memory usage snapshot."""
            self.memory_snapshots[name] = memory_profiler.get_memory_usage()
            
        def calculate_throughput(self, operation: str, item_count: int) -> float:
            """Calculate throughput (items per second)."""
            duration = self.get_duration(operation)
            if duration == 0:
                return 0.0
            return item_count / duration
            
        def assert_performance_threshold(self, operation: str, max_duration: float):
            """Assert that operation completed within threshold."""
            duration = self.get_duration(operation)
            assert duration <= max_duration, (
                f"Operation '{operation}' took {duration:.3f}s, "
                f"exceeding threshold of {max_duration}s"
            )
            
        def assert_throughput_threshold(self, operation: str, item_count: int, min_throughput: float):
            """Assert that throughput meets minimum threshold."""
            throughput = self.calculate_throughput(operation, item_count)
            assert throughput >= min_throughput, (
                f"Operation '{operation}' throughput {throughput:.2f} items/s "
                f"below threshold of {min_throughput} items/s"
            )
            
        def get_performance_report(self) -> Dict[str, Any]:
            """Generate comprehensive performance report."""
            report = {
                "timers": {},
                "counters": self.counters.copy(),
                "memory_snapshots": self.memory_snapshots.copy(),
                "summary": {}
            }
            
            for operation, data in self.timers.items():
                if "duration" in data:
                    report["timers"][operation] = {
                        "duration_seconds": data["duration"],
                        "duration_ms": data["duration"] * 1000
                    }
            
            # Calculate summary statistics
            if report["timers"]:
                durations = [t["duration_seconds"] for t in report["timers"].values()]
                report["summary"] = {
                    "total_operations": len(durations),
                    "total_time": sum(durations),
                    "average_time": sum(durations) / len(durations),
                    "min_time": min(durations),
                    "max_time": max(durations)
                }
                
            return report
    
    return PerformanceMonitor()


@pytest.fixture
def load_test_scenarios():
    """Different load testing scenarios."""
    return {
        "light_load": {
            "concurrent_users": 1,
            "emails_per_user": 10,
            "delay_between_requests": 0.1
        },
        "moderate_load": {
            "concurrent_users": 5,
            "emails_per_user": 20,
            "delay_between_requests": 0.05
        },
        "heavy_load": {
            "concurrent_users": 10,
            "emails_per_user": 50,
            "delay_between_requests": 0.01
        },
        "stress_test": {
            "concurrent_users": 20,
            "emails_per_user": 100,
            "delay_between_requests": 0.001
        }
    }


@pytest.fixture
def performance_assertions():
    """Helper functions for performance assertions."""
    class PerformanceAssertions:
        @staticmethod
        def assert_response_time(duration: float, max_seconds: float, operation: str = ""):
            """Assert response time is within acceptable limits."""
            assert duration <= max_seconds, (
                f"Response time for {operation} was {duration:.3f}s, "
                f"exceeding limit of {max_seconds}s"
            )
            
        @staticmethod
        def assert_memory_usage(current_mb: float, max_mb: float, operation: str = ""):
            """Assert memory usage is within acceptable limits."""
            assert current_mb <= max_mb, (
                f"Memory usage for {operation} was {current_mb:.2f}MB, "
                f"exceeding limit of {max_mb}MB"
            )
            
        @staticmethod
        def assert_cpu_usage(current_percent: float, max_percent: float, operation: str = ""):
            """Assert CPU usage is within acceptable limits."""
            assert current_percent <= max_percent, (
                f"CPU usage for {operation} was {current_percent:.1f}%, "
                f"exceeding limit of {max_percent}%"
            )
            
        @staticmethod
        def assert_throughput(actual: float, minimum: float, operation: str = ""):
            """Assert throughput meets minimum requirements."""
            assert actual >= minimum, (
                f"Throughput for {operation} was {actual:.2f} items/s, "
                f"below minimum of {minimum} items/s"
            )
            
        @staticmethod
        def assert_no_memory_leaks(before_mb: float, after_mb: float, tolerance_mb: float = 5.0):
            """Assert no significant memory leaks occurred."""
            memory_increase = after_mb - before_mb
            assert memory_increase <= tolerance_mb, (
                f"Memory increased by {memory_increase:.2f}MB, "
                f"exceeding leak tolerance of {tolerance_mb}MB"
            )
            
        @staticmethod
        def assert_linear_scaling(single_item_time: float, batch_size: int, actual_batch_time: float, tolerance: float = 0.2):
            """Assert that batch processing scales approximately linearly."""
            expected_time = single_item_time * batch_size
            efficiency = actual_batch_time / expected_time
            
            # Allow for some overhead but should be reasonably efficient
            assert efficiency <= (1.0 + tolerance), (
                f"Batch processing efficiency {efficiency:.2f} indicates poor scaling. "
                f"Expected ~{expected_time:.2f}s, actual {actual_batch_time:.2f}s"
            )
    
    return PerformanceAssertions()


# Performance test markers
performance_slow = pytest.mark.slow
performance_benchmark = pytest.mark.performance
performance_memory = pytest.mark.parametrize("memory_limit", [64, 128, 256, 512])
performance_batch_sizes = pytest.mark.parametrize("batch_size", [1, 10, 50, 100, 500])
performance_concurrent = pytest.mark.parametrize("concurrent_workers", [1, 2, 4, 8])


def pytest_configure(config):
    """Configure performance test markers."""
    config.addinivalue_line(
        "markers", "performance_slow: Slow performance tests that may take several minutes"
    )
    config.addinivalue_line(
        "markers", "performance_benchmark: Benchmark tests for measuring baseline performance"
    )
    config.addinivalue_line(
        "markers", "performance_memory: Memory usage performance tests"
    )
    config.addinivalue_line(
        "markers", "performance_load: Load testing scenarios"
    )
    config.addinivalue_line(
        "markers", "performance_stress: Stress testing scenarios"
    )
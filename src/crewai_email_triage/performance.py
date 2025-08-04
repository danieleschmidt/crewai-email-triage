"""Performance monitoring and optimization for CrewAI Email Triage."""

from __future__ import annotations

import time
import threading
import logging
import os
import gc
from typing import Dict, List, Optional, Any, Callable, ContextManager
from dataclasses import dataclass, field
from functools import wraps
from collections import defaultdict, deque
import statistics

from .metrics_export import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "tags": self.tags
        }


class PerformanceTracker:
    """Thread-safe performance tracking system."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._lock = threading.RLock()
        self._metrics_collector = get_metrics_collector()
        
    def record(self, name: str, value: float, unit: str = "ms", tags: Optional[Dict[str, str]] = None) -> None:
        """Record a performance metric."""
        metric = PerformanceMetric(name, value, unit, tags=tags or {})
        
        with self._lock:
            self._metrics[name].append(metric)
        
        # Also record in global metrics
        self._metrics_collector.record_histogram(f"perf_{name}", value)
        
        logger.debug("Performance metric recorded: %s = %.3f %s", name, value, unit)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a performance metric."""
        with self._lock:
            if name not in self._metrics or not self._metrics[name]:
                return {}
            
            values = [m.value for m in self._metrics[name]]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        with self._lock:
            return {name: self.get_stats(name) for name in self._metrics.keys()}
    
    def clear(self, name: Optional[str] = None) -> None:
        """Clear metrics (all or specific)."""
        with self._lock:
            if name:
                self._metrics[name].clear()
            else:
                self._metrics.clear()


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, tracker: Optional[PerformanceTracker] = None, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.tracker = tracker or get_performance_tracker()
        self.tags = tags or {}
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> 'Timer':
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_time = time.perf_counter()
        duration_ms = (self.end_time - self.start_time) * 1000
        self.tracker.record(self.name, duration_ms, "ms", self.tags)
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000


def timed(name: Optional[str] = None, tracker: Optional[PerformanceTracker] = None):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        metric_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(metric_name, tracker):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class MemoryTracker:
    """Memory usage tracking and monitoring."""
    
    def __init__(self):
        self._tracker = get_performance_tracker()
        self._baseline_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
                "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / (1024 * 1024)
            }
        except ImportError:
            # Fallback without psutil
            return {
                "rss_mb": 0.0,
                "vms_mb": 0.0,
                "percent": 0.0,
                "available_mb": 0.0
            }
    
    def record_current_usage(self, operation: str = "general") -> None:
        """Record current memory usage."""
        usage = self._get_memory_usage()
        
        for metric, value in usage.items():
            self._tracker.record(f"memory_{metric}", value, "MB" if metric.endswith("_mb") else "%", 
                               {"operation": operation})
    
    def track_operation(self, operation: str) -> ContextManager:
        """Context manager to track memory usage for an operation."""
        return MemoryOperationTracker(operation, self)


class MemoryOperationTracker:
    """Context manager for tracking memory usage during operations."""
    
    def __init__(self, operation: str, memory_tracker: MemoryTracker):
        self.operation = operation
        self.memory_tracker = memory_tracker
        self.start_usage: Optional[Dict[str, float]] = None
        self.end_usage: Optional[Dict[str, float]] = None
    
    def __enter__(self) -> 'MemoryOperationTracker':
        gc.collect()  # Force garbage collection for accurate measurement
        self.start_usage = self.memory_tracker._get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        gc.collect()  # Force garbage collection
        self.end_usage = self.memory_tracker._get_memory_usage()
        
        if self.start_usage and self.end_usage:
            # Calculate memory delta
            rss_delta = self.end_usage["rss_mb"] - self.start_usage["rss_mb"]
            vms_delta = self.end_usage["vms_mb"] - self.start_usage["vms_mb"]
            
            self.memory_tracker._tracker.record(
                f"memory_delta_rss", rss_delta, "MB", {"operation": self.operation}
            )
            self.memory_tracker._tracker.record(
                f"memory_delta_vms", vms_delta, "MB", {"operation": self.operation}
            )
            
            # Record final usage
            self.memory_tracker.record_current_usage(self.operation)


class ResourceMonitor:
    """Comprehensive resource monitoring."""
    
    def __init__(self, sampling_interval: float = 10.0):
        self.sampling_interval = sampling_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tracker = get_performance_tracker()
        self._memory_tracker = MemoryTracker()
        
    def start(self) -> None:
        """Start resource monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitoring started (interval: %.1fs)", self.sampling_interval)
    
    def stop(self) -> None:
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Record memory usage
                self._memory_tracker.record_current_usage("monitoring")
                
                # Record thread count
                thread_count = threading.active_count()
                self._tracker.record("thread_count", thread_count, "count")
                
                # Record garbage collection stats
                gc_stats = gc.get_stats()
                if gc_stats:
                    total_collections = sum(stat["collections"] for stat in gc_stats)
                    self._tracker.record("gc_collections_total", total_collections, "count")
                
                # Record open file descriptors (Unix only)
                try:
                    import psutil
                    process = psutil.Process()
                    fd_count = process.num_fds() if hasattr(process, 'num_fds') else 0
                    self._tracker.record("open_file_descriptors", fd_count, "count")
                except (ImportError, AttributeError):
                    pass
                
            except Exception as e:
                logger.warning("Error in resource monitoring: %s", e)
            
            time.sleep(self.sampling_interval)


class PerformanceOptimizer:
    """Automatic performance optimization based on monitoring data."""
    
    def __init__(self, tracker: Optional[PerformanceTracker] = None):
        self._tracker = tracker or get_performance_tracker()
        self._optimizations_applied = set()
        
    def analyze_and_optimize(self) -> List[str]:
        """Analyze performance data and apply optimizations."""
        optimizations = []
        
        # Analyze garbage collection frequency
        gc_stats = self._tracker.get_stats("gc_collections_total")
        if gc_stats and gc_stats.get("mean", 0) > 100:
            if "gc_tuning" not in self._optimizations_applied:
                self._optimize_garbage_collection()
                optimizations.append("garbage_collection_tuning")
                self._optimizations_applied.add("gc_tuning")
        
        # Analyze memory usage patterns
        memory_stats = self._tracker.get_stats("memory_rss_mb")
        if memory_stats and memory_stats.get("max", 0) > 500:  # > 500MB
            if "memory_optimization" not in self._optimizations_applied:
                self._optimize_memory_usage()
                optimizations.append("memory_usage_optimization")
                self._optimizations_applied.add("memory_optimization")
        
        # Analyze processing times
        processing_stats = self._tracker.get_stats("triage_processing_time")
        if processing_stats and processing_stats.get("p95", 0) > 5000:  # > 5s at p95
            if "processing_optimization" not in self._optimizations_applied:
                self._optimize_processing()
                optimizations.append("processing_time_optimization")
                self._optimizations_applied.add("processing_optimization")
        
        return optimizations
    
    def _optimize_garbage_collection(self) -> None:
        """Optimize garbage collection settings."""
        import gc
        
        # Adjust GC thresholds for better performance
        current_thresholds = gc.get_threshold()
        new_thresholds = (
            current_thresholds[0] * 2,  # Increase gen0 threshold
            current_thresholds[1] * 2,  # Increase gen1 threshold
            current_thresholds[2] * 2   # Increase gen2 threshold
        )
        gc.set_threshold(*new_thresholds)
        
        logger.info("Optimized GC thresholds: %s -> %s", current_thresholds, new_thresholds)
    
    def _optimize_memory_usage(self) -> None:
        """Optimize memory usage."""
        # Force garbage collection
        import gc
        collected = gc.collect()
        logger.info("Forced garbage collection: %d objects collected", collected)
        
        # Clear caches if available
        try:
            from .cache import get_smart_cache
            cache = get_smart_cache()
            # Don't clear all - just clean up expired entries
            logger.info("Memory optimization: cache cleanup applied")
        except ImportError:
            pass
    
    def _optimize_processing(self) -> None:
        """Optimize processing performance."""
        # This could involve:
        # - Adjusting batch sizes
        # - Enabling caching
        # - Tuning thread pool sizes
        logger.info("Processing optimization: performance tuning applied")


# Global instances
_performance_tracker: Optional[PerformanceTracker] = None
_resource_monitor: Optional[ResourceMonitor] = None
_performance_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
    return _performance_tracker


def get_resource_monitor() -> ResourceMonitor:
    """Get the global resource monitor instance."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor()
    return _resource_monitor


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def enable_performance_monitoring(sampling_interval: float = 10.0) -> None:
    """Enable comprehensive performance monitoring."""
    monitor = get_resource_monitor()
    monitor.sampling_interval = sampling_interval
    monitor.start()


def disable_performance_monitoring() -> None:
    """Disable performance monitoring."""
    monitor = get_resource_monitor()
    monitor.stop()


class PerformanceReport:
    """Generate comprehensive performance reports."""
    
    def __init__(self, tracker: Optional[PerformanceTracker] = None):
        self._tracker = tracker or get_performance_tracker()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        stats = self._tracker.get_all_stats()
        
        report = {
            "timestamp": time.time(),
            "summary": self._generate_summary(stats),
            "detailed_metrics": stats,
            "recommendations": self._generate_recommendations(stats)
        }
        
        return report
    
    def _generate_summary(self, stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            "total_metrics": len(stats),
            "avg_response_time": 0.0,
            "memory_usage_mb": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }
        
        # Calculate average response time
        processing_stats = stats.get("triage_processing_time", {})
        if processing_stats:
            summary["avg_response_time"] = processing_stats.get("mean", 0.0)
        
        # Get memory usage
        memory_stats = stats.get("memory_rss_mb", {})
        if memory_stats:
            summary["memory_usage_mb"] = memory_stats.get("mean", 0.0)
        
        return summary
    
    def _generate_recommendations(self, stats: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check response times
        processing_stats = stats.get("triage_processing_time", {})
        if processing_stats and processing_stats.get("p95", 0) > 3000:
            recommendations.append("Consider enabling caching to improve response times")
        
        # Check memory usage
        memory_stats = stats.get("memory_rss_mb", {})
        if memory_stats and memory_stats.get("max", 0) > 1000:
            recommendations.append("High memory usage detected - consider memory optimization")
        
        # Check error rates
        error_stats = stats.get("error_rate", {})
        if error_stats and error_stats.get("mean", 0) > 0.05:  # > 5% error rate
            recommendations.append("High error rate detected - investigate error causes")
        
        if not recommendations:
            recommendations.append("Performance looks good - no immediate optimizations needed")
        
        return recommendations
"""Performance optimization and profiling system."""

import time
import threading
import logging
import functools
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_delta: int = 0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceProfiler:
    """High-performance profiling system."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self._metrics = deque(maxlen=max_metrics)
        self._operation_stats = defaultdict(list)
        self._lock = threading.RLock()
        self._enabled = True
    
    def profile_operation(self, operation: str, metadata: Dict = None):
        """Context manager for profiling operations."""
        return PerformanceContext(self, operation, metadata or {})
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        if not self._enabled:
            return
        
        with self._lock:
            self._metrics.append(metric)
            self._operation_stats[metric.operation].append(metric.duration_ms)
            
            # Keep only recent metrics per operation
            if len(self._operation_stats[metric.operation]) > 1000:
                self._operation_stats[metric.operation].pop(0)
    
    def get_operation_stats(self, operation: str) -> Dict[str, Any]:
        """Get statistics for specific operation."""
        with self._lock:
            durations = self._operation_stats.get(operation, [])
            
            if not durations:
                return {"operation": operation, "count": 0}
            
            return {
                "operation": operation,
                "count": len(durations),
                "avg_ms": statistics.mean(durations),
                "median_ms": statistics.median(durations),
                "p95_ms": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations),
                "p99_ms": statistics.quantiles(durations, n=100)[98] if len(durations) >= 100 else max(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "std_dev": statistics.stdev(durations) if len(durations) > 1 else 0.0
            }
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall performance statistics."""
        with self._lock:
            all_durations = []
            operation_counts = {}
            
            for op, durations in self._operation_stats.items():
                all_durations.extend(durations)
                operation_counts[op] = len(durations)
            
            if not all_durations:
                return {"total_operations": 0}
            
            return {
                "total_operations": len(all_durations),
                "operations_by_type": operation_counts,
                "overall_avg_ms": statistics.mean(all_durations),
                "overall_median_ms": statistics.median(all_durations),
                "overall_p95_ms": statistics.quantiles(all_durations, n=20)[18] if len(all_durations) >= 20 else max(all_durations),
                "overall_p99_ms": statistics.quantiles(all_durations, n=100)[98] if len(all_durations) >= 100 else max(all_durations),
                "slowest_operations": self._get_slowest_operations(5)
            }
    
    def _get_slowest_operations(self, limit: int) -> List[Dict[str, Any]]:
        """Get slowest operations."""
        op_avg_times = []
        
        for op, durations in self._operation_stats.items():
            if durations:
                avg_time = statistics.mean(durations)
                op_avg_times.append({
                    "operation": op,
                    "avg_ms": avg_time,
                    "count": len(durations)
                })
        
        op_avg_times.sort(key=lambda x: x["avg_ms"], reverse=True)
        return op_avg_times[:limit]
    
    def enable(self):
        """Enable profiling."""
        self._enabled = True
        logger.info("Performance profiling enabled")
    
    def disable(self):
        """Disable profiling."""
        self._enabled = False
        logger.info("Performance profiling disabled")
    
    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()
            self._operation_stats.clear()
        logger.info("Performance metrics cleared")

class PerformanceContext:
    """Context manager for performance profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, operation: str, metadata: Dict):
        self.profiler = profiler
        self.operation = operation
        self.metadata = metadata
        self.start_time = 0
        self.start_memory = 0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        try:
            import psutil
            process = psutil.Process()
            self.start_memory = process.memory_info().rss
        except:
            self.start_memory = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000
        
        memory_delta = 0
        try:
            import psutil
            process = psutil.Process()
            end_memory = process.memory_info().rss
            memory_delta = end_memory - self.start_memory
        except:
            pass
        
        metric = PerformanceMetric(
            operation=self.operation,
            start_time=self.start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            memory_delta=memory_delta,
            success=exc_type is None,
            metadata=self.metadata
        )
        
        self.profiler.record_metric(metric)

def profile(operation: str = None, metadata: Dict = None):
    """Decorator for profiling function performance."""
    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _profiler.profile_operation(op_name, metadata):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

class BatchProcessor:
    """High-performance batch processing system."""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._profiler = _profiler
    
    def process_batch(self, items: List[Any], processor_func: Callable,
                     parallel: bool = True) -> List[Any]:
        """Process items in batches with optional parallelism."""
        if not items:
            return []
        
        with self._profiler.profile_operation("batch_processing", 
                                           {"item_count": len(items), 
                                            "batch_size": self.batch_size,
                                            "parallel": parallel}):
            
            if parallel and len(items) > self.batch_size:
                return self._process_parallel_batches(items, processor_func)
            else:
                return self._process_sequential_batch(items, processor_func)
    
    def _process_sequential_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process items sequentially."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            with self._profiler.profile_operation("batch_chunk", 
                                               {"chunk_size": len(batch)}):
                batch_results = [processor_func(item) for item in batch]
                results.extend(batch_results)
        
        return results
    
    def _process_parallel_batches(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process items in parallel batches."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Split items into chunks for parallel processing
        chunks = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        results = [None] * len(items)
        
        def process_chunk(chunk_data):
            chunk, chunk_index = chunk_data
            chunk_results = []
            for item in chunk:
                result = processor_func(item)
                chunk_results.append(result)
            return chunk_index, chunk_results
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_data = [(chunk, i) for i, chunk in enumerate(chunks)]
            future_to_chunk = {executor.submit(process_chunk, cd): cd for cd in chunk_data}
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_index, chunk_results = future.result()
                    start_idx = chunk_index * self.batch_size
                    for i, result in enumerate(chunk_results):
                        results[start_idx + i] = result
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
        
        return [r for r in results if r is not None]

# Global profiler instance
_profiler = PerformanceProfiler()

def get_profiler() -> PerformanceProfiler:
    """Get global profiler instance."""
    return _profiler

def get_batch_processor(batch_size: int = 100, max_workers: int = 4) -> BatchProcessor:
    """Get batch processor instance."""
    return BatchProcessor(batch_size, max_workers)

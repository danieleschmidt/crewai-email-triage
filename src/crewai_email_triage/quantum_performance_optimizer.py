"""
Quantum Performance Optimizer v3.0
Advanced scaling, adaptive optimization, and self-improving patterns
"""

import time
import asyncio
import threading
import multiprocessing
import concurrent.futures
import queue
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import logging
from pathlib import Path
import psutil
import weakref

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    QUANTUM = "quantum"


class ScalingStrategy(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    timestamp: float
    throughput_mps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    active_workers: int
    queue_depth: int
    error_rate: float
    cache_hit_rate: float


@dataclass
class OptimizationRule:
    name: str
    trigger_condition: Callable[[PerformanceMetrics], bool]
    optimization_action: Callable[[], bool]
    cooldown_seconds: float
    priority: int
    last_executed: float = 0.0
    execution_count: int = 0
    success_rate: float = 0.0


@dataclass
class AdaptiveConfiguration:
    batch_size: int
    worker_count: int
    cache_size: int
    prefetch_count: int
    timeout_seconds: float
    enable_compression: bool
    enable_vectorization: bool
    optimization_level: OptimizationLevel
    scaling_strategy: ScalingStrategy


class IntelligentWorkloadManager:
    """Manages dynamic workload distribution and resource allocation."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.worker_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.performance_history: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        self._shutdown = False
        
    def initialize(self):
        """Initialize the workload manager."""
        if self.worker_pool is None:
            self.worker_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="quantum_worker"
            )
            logger.info(f"Workload manager initialized with {self.max_workers} workers")
    
    def submit_task(self, task_id: str, func: Callable, *args, priority: int = 5, **kwargs) -> concurrent.futures.Future:
        """Submit a task with priority scheduling."""
        if not self.worker_pool:
            self.initialize()
        
        # Wrap the function to track performance
        def tracked_func(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                with self._lock:
                    self.completed_tasks.append({
                        'task_id': task_id,
                        'execution_time': execution_time,
                        'success': True,
                        'timestamp': time.time()
                    })
                
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                
                with self._lock:
                    self.completed_tasks.append({
                        'task_id': task_id,
                        'execution_time': execution_time,
                        'success': False,
                        'error': str(e),
                        'timestamp': time.time()
                    })
                raise
        
        future = self.worker_pool.submit(tracked_func, *args, **kwargs)
        
        with self._lock:
            self.active_tasks[task_id] = {
                'future': future,
                'priority': priority,
                'submitted_at': time.time()
            }
        
        return future
    
    def process_batch_adaptive(self, items: List[Any], processor_func: Callable, 
                              batch_size: Optional[int] = None) -> List[Any]:
        """Process a batch with adaptive optimization."""
        if not items:
            return []
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(len(items))
        
        # Split into batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Submit batch processing tasks
        futures = []
        for i, batch in enumerate(batches):
            future = self.submit_task(
                task_id=f"batch_{i}",
                func=self._process_batch_with_monitoring,
                batch=batch,
                processor_func=processor_func,
                priority=1  # High priority for batch processing
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
        
        return results
    
    def _process_batch_with_monitoring(self, batch: List[Any], processor_func: Callable) -> List[Any]:
        """Process a batch while monitoring performance."""
        start_time = time.time()
        results = []
        
        for item in batch:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Item processing failed: {e}")
                results.append(None)
        
        processing_time = time.time() - start_time
        throughput = len(batch) / processing_time if processing_time > 0 else 0
        
        # Record performance metrics
        self._record_performance_metric(throughput, processing_time, len(batch))
        
        return results
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on historical performance."""
        if not self.performance_history:
            # Default batch size based on system resources
            return min(50, max(10, total_items // self.max_workers))
        
        # Analyze historical performance to find optimal batch size
        recent_metrics = list(self.performance_history)[-20:]  # Last 20 measurements
        
        if recent_metrics:
            avg_throughput = statistics.mean([m['throughput'] for m in recent_metrics])
            
            # Calculate batch size that maximizes throughput while minimizing latency
            if avg_throughput > 100:  # High throughput system
                optimal_batch = min(100, total_items // max(1, self.max_workers - 2))
            elif avg_throughput > 50:  # Medium throughput
                optimal_batch = min(50, total_items // max(1, self.max_workers))
            else:  # Low throughput
                optimal_batch = min(20, total_items // max(1, self.max_workers + 2))
            
            return max(1, optimal_batch)
        
        return min(25, max(5, total_items // self.max_workers))
    
    def _record_performance_metric(self, throughput: float, processing_time: float, batch_size: int):
        """Record performance metrics for optimization."""
        metric = {
            'timestamp': time.time(),
            'throughput': throughput,
            'processing_time': processing_time,
            'batch_size': batch_size,
            'cpu_usage': psutil.cpu_percent(),
            'memory_mb': psutil.Process().memory_info().rss / (1024 * 1024)
        }
        
        self.performance_history.append(metric)
    
    def get_workload_status(self) -> Dict[str, Any]:
        """Get current workload status."""
        with self._lock:
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)
            
            recent_completions = [t for t in self.completed_tasks 
                                if time.time() - t['timestamp'] < 300]  # Last 5 minutes
            
            success_rate = (
                sum(1 for t in recent_completions if t['success']) / len(recent_completions)
                if recent_completions else 1.0
            )
            
            avg_execution_time = (
                statistics.mean([t['execution_time'] for t in recent_completions])
                if recent_completions else 0.0
            )
            
            return {
                'active_tasks': active_count,
                'completed_tasks': completed_count,
                'success_rate': success_rate,
                'avg_execution_time_ms': avg_execution_time * 1000,
                'worker_count': self.max_workers,
                'performance_samples': len(self.performance_history)
            }
    
    def shutdown(self):
        """Shutdown the workload manager."""
        self._shutdown = True
        if self.worker_pool:
            self.worker_pool.shutdown(wait=True)
            logger.info("Workload manager shutdown complete")


class AdaptiveCache:
    """Self-optimizing cache with intelligent eviction and prefetching."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.hit_counts: Dict[str, int] = defaultdict(int)
        self.miss_counts: Dict[str, int] = defaultdict(int)
        self.size_bytes: int = 0
        self._lock = threading.RLock()
        
        # Adaptive parameters
        self.prefetch_threshold = 3  # Access count to trigger prefetch
        self.eviction_batch_size = max(10, max_size // 10)
        
    def get(self, key: str) -> Tuple[Optional[Any], bool]:
        """Get value from cache, returning (value, hit)."""
        with self._lock:
            if key in self.cache:
                self.hit_counts[key] += 1
                self.access_history[key].append(time.time())
                return self.cache[key], True
            else:
                self.miss_counts[key] += 1
                return None, False
    
    def put(self, key: str, value: Any, size_hint: Optional[int] = None):
        """Put value in cache with intelligent eviction."""
        with self._lock:
            # Estimate size if not provided
            if size_hint is None:
                size_hint = self._estimate_size(value)
            
            # Check if we need to evict
            if len(self.cache) >= self.max_size:
                self._intelligent_eviction()
            
            self.cache[key] = value
            self.access_history[key].append(time.time())
            self.size_bytes += size_hint
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value in bytes."""
        try:
            import sys
            return sys.getsizeof(value)
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) 
                          for k, v in value.items())
            else:
                return 64  # Default estimate
    
    def _intelligent_eviction(self):
        """Intelligently evict least valuable items."""
        if not self.cache:
            return
        
        # Calculate value score for each item
        current_time = time.time()
        scores = {}
        
        for key in self.cache:
            # Factors: access frequency, recency, hit ratio
            hit_ratio = self.hit_counts[key] / (self.hit_counts[key] + self.miss_counts[key])
            recent_accesses = sum(1 for t in self.access_history[key] 
                                if current_time - t < 3600)  # Last hour
            last_access = max(self.access_history[key]) if self.access_history[key] else 0
            recency_score = max(0, 1 - (current_time - last_access) / 3600)  # Decay over 1 hour
            
            # Combined score (higher is better)
            scores[key] = hit_ratio * 0.4 + recent_accesses * 0.3 + recency_score * 0.3
        
        # Sort by score and evict lowest scoring items
        sorted_items = sorted(scores.items(), key=lambda x: x[1])
        evict_count = min(self.eviction_batch_size, len(sorted_items))
        
        for key, _ in sorted_items[:evict_count]:
            if key in self.cache:
                del self.cache[key]
                # Clean up tracking data
                if key in self.access_history:
                    del self.access_history[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_hits = sum(self.hit_counts.values())
            total_misses = sum(self.miss_counts.values())
            hit_rate = total_hits / (total_hits + total_misses) if total_hits + total_misses > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'size_bytes': self.size_bytes,
                'avg_size_per_item': self.size_bytes / len(self.cache) if self.cache else 0
            }


class QuantumOptimizationEngine:
    """Advanced optimization engine with self-learning capabilities."""
    
    def __init__(self):
        self.configuration = AdaptiveConfiguration(
            batch_size=25,
            worker_count=multiprocessing.cpu_count(),
            cache_size=1000,
            prefetch_count=10,
            timeout_seconds=30.0,
            enable_compression=False,
            enable_vectorization=True,
            optimization_level=OptimizationLevel.BALANCED,
            scaling_strategy=ScalingStrategy.ADAPTIVE
        )
        
        self.workload_manager = IntelligentWorkloadManager(self.configuration.worker_count)
        self.adaptive_cache = AdaptiveCache(self.configuration.cache_size)
        self.optimization_rules: List[OptimizationRule] = []
        self.metrics_history: deque = deque(maxlen=500)
        self._optimization_thread: Optional[threading.Thread] = None
        self._running = False
        
        self._setup_optimization_rules()
    
    def _setup_optimization_rules(self):
        """Setup intelligent optimization rules."""
        
        # Rule 1: Scale workers based on queue depth
        def high_queue_depth_trigger(metrics: PerformanceMetrics) -> bool:
            return metrics.queue_depth > self.configuration.worker_count * 2
        
        def scale_up_workers() -> bool:
            if self.configuration.worker_count < multiprocessing.cpu_count() * 2:
                self.configuration.worker_count += 1
                self.workload_manager.max_workers = self.configuration.worker_count
                logger.info(f"Scaled up workers to {self.configuration.worker_count}")
                return True
            return False
        
        self.optimization_rules.append(OptimizationRule(
            name="scale_up_workers",
            trigger_condition=high_queue_depth_trigger,
            optimization_action=scale_up_workers,
            cooldown_seconds=60.0,
            priority=1
        ))
        
        # Rule 2: Optimize batch size based on throughput
        def low_throughput_trigger(metrics: PerformanceMetrics) -> bool:
            if len(self.metrics_history) < 5:
                return False
            recent_throughput = [m.throughput_mps for m in list(self.metrics_history)[-5:]]
            avg_throughput = statistics.mean(recent_throughput)
            return avg_throughput < 10  # Low throughput threshold
        
        def increase_batch_size() -> bool:
            if self.configuration.batch_size < 100:
                self.configuration.batch_size = min(100, int(self.configuration.batch_size * 1.5))
                logger.info(f"Increased batch size to {self.configuration.batch_size}")
                return True
            return False
        
        self.optimization_rules.append(OptimizationRule(
            name="optimize_batch_size",
            trigger_condition=low_throughput_trigger,
            optimization_action=increase_batch_size,
            cooldown_seconds=120.0,
            priority=2
        ))
        
        # Rule 3: Enable compression for high memory usage
        def high_memory_trigger(metrics: PerformanceMetrics) -> bool:
            return metrics.memory_usage_mb > 1000 and not self.configuration.enable_compression
        
        def enable_compression() -> bool:
            self.configuration.enable_compression = True
            logger.info("Enabled compression due to high memory usage")
            return True
        
        self.optimization_rules.append(OptimizationRule(
            name="enable_compression",
            trigger_condition=high_memory_trigger,
            optimization_action=enable_compression,
            cooldown_seconds=300.0,
            priority=3
        ))
        
        # Rule 4: Adaptive cache sizing
        def cache_performance_trigger(metrics: PerformanceMetrics) -> bool:
            return metrics.cache_hit_rate < 0.7 and self.configuration.cache_size < 5000
        
        def increase_cache_size() -> bool:
            old_size = self.configuration.cache_size
            self.configuration.cache_size = min(5000, int(self.configuration.cache_size * 1.3))
            self.adaptive_cache.max_size = self.configuration.cache_size
            logger.info(f"Increased cache size from {old_size} to {self.configuration.cache_size}")
            return True
        
        self.optimization_rules.append(OptimizationRule(
            name="adaptive_cache_sizing",
            trigger_condition=cache_performance_trigger,
            optimization_action=increase_cache_size,
            cooldown_seconds=180.0,
            priority=2
        ))
    
    def start_optimization_loop(self):
        """Start the continuous optimization loop."""
        if self._running:
            return
        
        self._running = True
        self.workload_manager.initialize()
        
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self._optimization_thread.start()
        logger.info("Quantum optimization engine started")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self._running:
            try:
                # Collect current metrics
                current_metrics = self._collect_current_metrics()
                self.metrics_history.append(current_metrics)
                
                # Apply optimization rules
                self._apply_optimization_rules(current_metrics)
                
                # Sleep before next optimization cycle
                time.sleep(30.0)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                time.sleep(30.0)
    
    def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        workload_status = self.workload_manager.get_workload_status()
        cache_stats = self.adaptive_cache.get_stats()
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.Process().memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # Calculate latency percentiles from recent history
        if len(self.metrics_history) >= 3:
            recent_metrics = list(self.metrics_history)[-10:]
            latencies = []
            for m in recent_metrics:
                if hasattr(m, 'latency_samples'):
                    latencies.extend(m.latency_samples)
            
            if latencies:
                p50 = np.percentile(latencies, 50)
                p95 = np.percentile(latencies, 95)
                p99 = np.percentile(latencies, 99)
            else:
                p50 = p95 = p99 = workload_status['avg_execution_time_ms']
        else:
            p50 = p95 = p99 = workload_status['avg_execution_time_ms']
        
        return PerformanceMetrics(
            timestamp=time.time(),
            throughput_mps=workload_status.get('throughput', 0.0),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_mb,
            active_workers=self.configuration.worker_count,
            queue_depth=workload_status['active_tasks'],
            error_rate=1.0 - workload_status['success_rate'],
            cache_hit_rate=cache_stats['hit_rate']
        )
    
    def _apply_optimization_rules(self, metrics: PerformanceMetrics):
        """Apply optimization rules based on current metrics."""
        current_time = time.time()
        
        # Sort rules by priority (lower number = higher priority)
        sorted_rules = sorted(self.optimization_rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            try:
                # Check cooldown
                if current_time - rule.last_executed < rule.cooldown_seconds:
                    continue
                
                # Check trigger condition
                if rule.trigger_condition(metrics):
                    logger.info(f"Applying optimization rule: {rule.name}")
                    
                    success = rule.optimization_action()
                    rule.last_executed = current_time
                    rule.execution_count += 1
                    
                    # Update success rate
                    if rule.execution_count == 1:
                        rule.success_rate = 1.0 if success else 0.0
                    else:
                        # Exponential moving average
                        alpha = 0.3
                        rule.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * rule.success_rate
                    
                    if success:
                        logger.info(f"Optimization rule {rule.name} applied successfully")
                        # Only apply one rule per cycle to avoid conflicts
                        break
                    else:
                        logger.warning(f"Optimization rule {rule.name} failed")
                        
            except Exception as e:
                logger.error(f"Error applying optimization rule {rule.name}: {e}")
    
    def optimize_batch_processing(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Optimize batch processing with all enhancements."""
        if not self._running:
            self.start_optimization_loop()
        
        start_time = time.time()
        
        # Use adaptive batch processing
        results = self.workload_manager.process_batch_adaptive(
            items, processor_func, self.configuration.batch_size
        )
        
        processing_time = time.time() - start_time
        throughput = len(items) / processing_time if processing_time > 0 else 0
        
        logger.info(f"Processed {len(items)} items in {processing_time:.2f}s "
                   f"(throughput: {throughput:.1f} items/s)")
        
        return results
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        current_metrics = self._collect_current_metrics() if self._running else None
        
        return {
            'configuration': asdict(self.configuration),
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'optimization_rules': [
                {
                    'name': rule.name,
                    'execution_count': rule.execution_count,
                    'success_rate': rule.success_rate,
                    'last_executed': rule.last_executed,
                    'priority': rule.priority
                }
                for rule in self.optimization_rules
            ],
            'workload_status': self.workload_manager.get_workload_status(),
            'cache_stats': self.adaptive_cache.get_stats(),
            'metrics_history_size': len(self.metrics_history),
            'optimization_running': self._running
        }
    
    def shutdown(self):
        """Shutdown the optimization engine."""
        self._running = False
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5.0)
        self.workload_manager.shutdown()
        logger.info("Quantum optimization engine shutdown complete")


# Global optimization engine instance
quantum_optimizer = QuantumOptimizationEngine()


def optimized_processing(func: Callable):
    """Decorator for optimized function execution."""
    def wrapper(*args, **kwargs):
        if not quantum_optimizer._running:
            quantum_optimizer.start_optimization_loop()
        return func(*args, **kwargs)
    return wrapper


def get_quantum_optimizer() -> QuantumOptimizationEngine:
    """Get the global quantum optimization engine."""
    return quantum_optimizer
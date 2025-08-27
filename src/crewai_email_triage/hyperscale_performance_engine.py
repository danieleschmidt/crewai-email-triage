"""HyperScale Performance Engine for Email Triage.

Advanced performance optimization with auto-scaling, intelligent caching, and distributed processing.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
import hashlib
import pickle
from collections import defaultdict, OrderedDict
from functools import wraps, lru_cache
import weakref
from contextlib import asynccontextmanager
import json
import multiprocessing
import queue
import statistics

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    CONSERVATIVE = "conservative"  # Slow scale-up, fast scale-down
    BALANCED = "balanced"         # Moderate scaling
    AGGRESSIVE = "aggressive"     # Fast scale-up, slow scale-down
    REACTIVE = "reactive"         # React to current load only
    PREDICTIVE = "predictive"     # Use prediction models


class CacheStrategy(Enum):
    """Caching strategies."""
    LRU = "lru"                   # Least Recently Used
    LFU = "lfu"                   # Least Frequently Used
    TTL = "ttl"                   # Time To Live
    ADAPTIVE = "adaptive"         # Adaptive based on access patterns
    HIERARCHICAL = "hierarchical" # Multi-level caching


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    requests_per_second: float = 0.0
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization_mb: float = 0.0
    cache_hit_rate: float = 0.0
    active_workers: int = 0
    queue_depth: int = 0
    error_rate: float = 0.0
    throughput_items_per_minute: float = 0.0


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    action: str  # scale_up, scale_down, maintain
    target_workers: int
    reason: str
    confidence: float
    predicted_load: float
    current_utilization: float


class AdaptiveCache:
    """Intelligent adaptive caching system."""
    
    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self._cache: Dict[str, Any] = OrderedDict()
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._ttl: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Adaptive parameters
        self._hit_rate_window = []
        self._access_pattern_scores: Dict[str, float] = {}
        self._size_penalties: Dict[str, float] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache with adaptive scoring."""
        with self._lock:
            if key in self._cache:
                # Check TTL if applicable
                if key in self._ttl and time.time() > self._ttl[key]:
                    self._remove_key(key)
                    self.misses += 1
                    return default
                
                # Update access patterns
                self._access_times[key] = time.time()
                self._access_counts[key] += 1
                self._update_access_pattern_score(key)
                
                # Move to end for LRU behavior
                value = self._cache[key]
                del self._cache[key]
                self._cache[key] = value
                
                self.hits += 1
                return value
            else:
                self.misses += 1
                return default
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None):
        """Set item in cache with intelligent eviction."""
        with self._lock:
            # Set TTL if provided
            if ttl_seconds:
                self._ttl[key] = time.time() + ttl_seconds
            
            # If key exists, update
            if key in self._cache:
                del self._cache[key]
            
            # Add new item
            self._cache[key] = value
            self._access_times[key] = time.time()
            self._access_counts[key] += 1
            
            # Calculate size penalty for large objects
            try:
                size_penalty = len(pickle.dumps(value)) / (1024 * 1024)  # MB
                self._size_penalties[key] = size_penalty
            except Exception:
                self._size_penalties[key] = 1.0  # Default penalty
            
            # Evict if necessary
            while len(self._cache) > self.max_size:
                self._evict_item()
    
    def _evict_item(self):
        """Intelligently evict an item based on strategy and adaptive scoring."""
        if not self._cache:
            return
        
        if self.strategy == CacheStrategy.ADAPTIVE:
            # Use adaptive scoring for eviction
            scores = {}
            current_time = time.time()
            
            for key in self._cache:
                # Base score components
                recency = current_time - self._access_times.get(key, 0)
                frequency = self._access_counts.get(key, 1)
                size_penalty = self._size_penalties.get(key, 1.0)
                pattern_score = self._access_pattern_scores.get(key, 0.5)
                
                # Combined adaptive score (lower is better for eviction)
                score = (recency / 3600) - (frequency / 10) + size_penalty - pattern_score
                scores[key] = score
            
            # Evict item with highest score (worst)
            key_to_evict = max(scores.items(), key=lambda x: x[1])[0]
            
        elif self.strategy == CacheStrategy.LRU:
            # Evict least recently used (first in OrderedDict)
            key_to_evict = next(iter(self._cache))
            
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            key_to_evict = min(self._access_counts.items(), key=lambda x: x[1])[0]
            
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired items first, then oldest
            current_time = time.time()
            expired_keys = [k for k, ttl in self._ttl.items() if current_time > ttl]
            
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                key_to_evict = min(self._access_times.items(), key=lambda x: x[1])[0]
        else:
            # Default to LRU
            key_to_evict = next(iter(self._cache))
        
        self._remove_key(key_to_evict)
        self.evictions += 1
    
    def _remove_key(self, key: str):
        """Remove key and associated metadata."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
        if key in self._access_counts:
            del self._access_counts[key]
        if key in self._ttl:
            del self._ttl[key]
        if key in self._access_pattern_scores:
            del self._access_pattern_scores[key]
        if key in self._size_penalties:
            del self._size_penalties[key]
    
    def _update_access_pattern_score(self, key: str):
        """Update adaptive access pattern score for key."""
        current_time = time.time()
        access_time = self._access_times.get(key, current_time)
        access_count = self._access_counts.get(key, 1)
        
        # Calculate temporal locality (recent accesses)
        time_since_access = current_time - access_time
        temporal_score = 1.0 / (1.0 + time_since_access / 3600)  # Decay over hour
        
        # Calculate frequency locality
        frequency_score = min(access_count / 100.0, 1.0)  # Cap at 100 accesses
        
        # Combined pattern score
        pattern_score = 0.6 * temporal_score + 0.4 * frequency_score
        self._access_pattern_scores[key] = pattern_score
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'hit_rate': hit_rate,
                'strategy': self.strategy.value
            }
    
    def clear(self):
        """Clear all cache data."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._access_counts.clear()
            self._ttl.clear()
            self._access_pattern_scores.clear()
            self._size_penalties.clear()


class LoadPredictor:
    """Predictive load analysis for auto-scaling."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.load_history: List[Tuple[float, float]] = []  # (timestamp, load)
        self.prediction_accuracy_history: List[float] = []
        self._lock = threading.Lock()
    
    def record_load(self, load_value: float):
        """Record current load measurement."""
        with self._lock:
            timestamp = time.time()
            self.load_history.append((timestamp, load_value))
            
            # Keep only recent history
            if len(self.load_history) > self.history_size:
                self.load_history = self.load_history[-self.history_size:]
    
    def predict_load(self, prediction_horizon_seconds: float = 300.0) -> Tuple[float, float]:
        """Predict future load with confidence score."""
        with self._lock:
            if len(self.load_history) < 10:
                return 0.0, 0.0  # Insufficient data
            
            # Simple trend analysis with weighted recent data
            recent_data = self.load_history[-50:]  # Last 50 measurements
            
            # Calculate weighted moving average with trend
            weights = [i + 1 for i in range(len(recent_data))]
            weighted_avg = sum(load * weight for (_, load), weight in zip(recent_data, weights)) / sum(weights)
            
            # Calculate trend (simple linear regression)
            if len(recent_data) >= 3:
                timestamps = [ts - recent_data[0][0] for ts, _ in recent_data]
                loads = [load for _, load in recent_data]
                
                n = len(recent_data)
                sum_x = sum(timestamps)
                sum_y = sum(loads)
                sum_xy = sum(x * y for x, y in zip(timestamps, loads))
                sum_x2 = sum(x * x for x in timestamps)
                
                # Linear regression slope
                denominator = n * sum_x2 - sum_x * sum_x
                if denominator != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / denominator
                    
                    # Project trend into future
                    last_timestamp = recent_data[-1][0]
                    future_timestamp = last_timestamp + prediction_horizon_seconds
                    time_delta = prediction_horizon_seconds
                    
                    predicted_load = weighted_avg + slope * time_delta
                    
                    # Calculate confidence based on trend consistency
                    variance = statistics.variance(loads) if len(loads) > 1 else 0
                    confidence = max(0.1, min(1.0 - variance / max(weighted_avg, 1.0), 0.9))
                    
                    return max(0.0, predicted_load), confidence
            
            return weighted_avg, 0.5  # Default confidence


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.BALANCED):
        self.strategy = strategy
        self.min_workers = 1
        self.max_workers = multiprocessing.cpu_count() * 2
        self.current_workers = self.min_workers
        
        self.load_predictor = LoadPredictor()
        self.scaling_history: List[Tuple[float, ScalingDecision]] = []
        
        # Scaling parameters by strategy
        self.scaling_params = {
            ScalingStrategy.CONSERVATIVE: {
                'scale_up_threshold': 0.8,
                'scale_down_threshold': 0.3,
                'scale_up_factor': 1.2,
                'scale_down_factor': 0.8,
                'cooldown_period': 300  # 5 minutes
            },
            ScalingStrategy.BALANCED: {
                'scale_up_threshold': 0.7,
                'scale_down_threshold': 0.4,
                'scale_up_factor': 1.5,
                'scale_down_factor': 0.7,
                'cooldown_period': 180  # 3 minutes
            },
            ScalingStrategy.AGGRESSIVE: {
                'scale_up_threshold': 0.6,
                'scale_down_threshold': 0.5,
                'scale_up_factor': 2.0,
                'scale_down_factor': 0.6,
                'cooldown_period': 60   # 1 minute
            }
        }
        
        self._last_scaling_time = 0.0
        self._lock = threading.Lock()
    
    def should_scale(self, current_metrics: PerformanceMetrics) -> ScalingDecision:
        """Determine if scaling action is needed."""
        with self._lock:
            current_time = time.time()
            params = self.scaling_params[self.strategy]
            
            # Check cooldown period
            if current_time - self._last_scaling_time < params['cooldown_period']:
                return ScalingDecision(
                    action="maintain",
                    target_workers=self.current_workers,
                    reason="Cooling down from recent scaling",
                    confidence=1.0,
                    predicted_load=0.0,
                    current_utilization=0.0
                )
            
            # Calculate current utilization
            utilization = self._calculate_utilization(current_metrics)
            
            # Record load for prediction
            self.load_predictor.record_load(utilization)
            
            # Get prediction for scaling decision
            predicted_load, confidence = self.load_predictor.predict_load(300.0)  # 5 minutes
            
            decision = ScalingDecision(
                action="maintain",
                target_workers=self.current_workers,
                reason="Within acceptable range",
                confidence=confidence,
                predicted_load=predicted_load,
                current_utilization=utilization
            )
            
            # Scaling logic
            if utilization > params['scale_up_threshold'] or predicted_load > params['scale_up_threshold']:
                # Scale up
                target_workers = min(
                    int(self.current_workers * params['scale_up_factor']),
                    self.max_workers
                )
                
                if target_workers > self.current_workers:
                    decision.action = "scale_up"
                    decision.target_workers = target_workers
                    decision.reason = f"High utilization: {utilization:.2f} (predicted: {predicted_load:.2f})"
                    
            elif utilization < params['scale_down_threshold'] and predicted_load < params['scale_down_threshold']:
                # Scale down
                target_workers = max(
                    int(self.current_workers * params['scale_down_factor']),
                    self.min_workers
                )
                
                if target_workers < self.current_workers:
                    decision.action = "scale_down"
                    decision.target_workers = target_workers
                    decision.reason = f"Low utilization: {utilization:.2f} (predicted: {predicted_load:.2f})"
            
            # Record decision
            self.scaling_history.append((current_time, decision))
            
            # Limit history size
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-1000:]
            
            return decision
    
    def execute_scaling(self, decision: ScalingDecision) -> bool:
        """Execute scaling decision."""
        if decision.action == "maintain":
            return True
        
        with self._lock:
            old_workers = self.current_workers
            self.current_workers = decision.target_workers
            self._last_scaling_time = time.time()
            
            logger.info(f"Auto-scaling: {decision.action} from {old_workers} to {decision.target_workers} workers. Reason: {decision.reason}")
            return True
    
    def _calculate_utilization(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall system utilization."""
        # Weighted utilization score
        cpu_weight = 0.4
        response_time_weight = 0.3
        queue_weight = 0.2
        error_weight = 0.1
        
        # Normalize metrics
        cpu_score = min(metrics.cpu_utilization / 100.0, 1.0)
        
        # Response time score (higher is worse)
        response_time_score = min(metrics.average_response_time_ms / 1000.0, 1.0)
        
        # Queue depth score
        max_queue_size = max(metrics.active_workers * 10, 100)
        queue_score = min(metrics.queue_depth / max_queue_size, 1.0)
        
        # Error rate score
        error_score = min(metrics.error_rate, 1.0)
        
        utilization = (cpu_score * cpu_weight + 
                      response_time_score * response_time_weight + 
                      queue_score * queue_weight + 
                      error_score * error_weight)
        
        return min(utilization, 1.0)


class HyperScalePerformanceEngine:
    """Complete hyperscale performance engine with all optimizations."""
    
    def __init__(self):
        # Core components
        self.cache = AdaptiveCache(max_size=50000, strategy=CacheStrategy.ADAPTIVE)
        self.autoscaler = AutoScaler(strategy=ScalingStrategy.BALANCED)
        
        # Thread pools with dynamic sizing
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._process_pool: Optional[ProcessPoolExecutor] = None
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self._metrics_history: List[Tuple[float, PerformanceMetrics]] = []
        self._request_times: List[float] = []
        
        # Configuration
        self.enable_process_pool = True
        self.enable_threading = True
        self.enable_caching = True
        self.enable_auto_scaling = True
        
        self._lock = threading.Lock()
        self._performance_monitor_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
    
    def start(self):
        """Start the performance engine."""
        logger.info("Starting HyperScale Performance Engine")
        
        # Initialize thread pool
        self._thread_pool = ThreadPoolExecutor(
            max_workers=self.autoscaler.current_workers,
            thread_name_prefix="hyperscale"
        )
        
        # Initialize process pool if enabled
        if self.enable_process_pool:
            try:
                self._process_pool = ProcessPoolExecutor(
                    max_workers=min(multiprocessing.cpu_count(), 4)
                )
            except Exception as e:
                logger.warning(f"Process pool initialization failed: {e}")
                self.enable_process_pool = False
        
        # Start performance monitoring
        if self.enable_auto_scaling:
            self._monitoring_active = True
            self._performance_monitor_thread = threading.Thread(
                target=self._performance_monitor_loop,
                daemon=True
            )
            self._performance_monitor_thread.start()
        
        logger.info("HyperScale Performance Engine started")
    
    def stop(self):
        """Stop the performance engine."""
        logger.info("Stopping HyperScale Performance Engine")
        
        # Stop monitoring
        self._monitoring_active = False
        if self._performance_monitor_thread:
            self._performance_monitor_thread.join(timeout=5.0)
        
        # Shutdown thread pools
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        
        if self._process_pool:
            self._process_pool.shutdown(wait=True)
        
        logger.info("HyperScale Performance Engine stopped")
    
    def process_email_optimized(self, content: str, processing_func: Callable, 
                               use_cache: bool = True, use_parallel: bool = True) -> Any:
        """Process email with all optimizations enabled."""
        start_time = time.time()
        
        try:
            # Check cache first
            if use_cache and self.enable_caching:
                cache_key = self._generate_cache_key(content, processing_func.__name__)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self._record_request_time(time.time() - start_time)
                    return cached_result
            
            # Process with optimal execution strategy
            if use_parallel and self.enable_threading:
                result = self._process_parallel(content, processing_func)
            else:
                result = processing_func(content)
            
            # Cache result
            if use_cache and self.enable_caching:
                cache_key = self._generate_cache_key(content, processing_func.__name__)
                # Set TTL based on content characteristics
                ttl = self._calculate_ttl(content, result)
                self.cache.set(cache_key, result, ttl)
            
            # Record timing
            self._record_request_time(time.time() - start_time)
            
            return result
            
        except Exception as e:
            # Record error timing
            self._record_request_time(time.time() - start_time, error=True)
            raise
    
    def process_batch_optimized(self, items: List[str], processing_func: Callable,
                               batch_size: int = None, use_cache: bool = True) -> List[Any]:
        """Process batch of emails with optimal parallelization."""
        start_time = time.time()
        
        if not items:
            return []
        
        # Determine optimal batch size
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(len(items))
        
        results = []
        cached_results = {}
        items_to_process = []
        
        # Check cache for all items
        if use_cache and self.enable_caching:
            for i, item in enumerate(items):
                cache_key = self._generate_cache_key(item, processing_func.__name__)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    cached_results[i] = cached_result
                else:
                    items_to_process.append((i, item))
        else:
            items_to_process = list(enumerate(items))
        
        # Process remaining items in parallel
        if items_to_process:
            if self.enable_threading and self._thread_pool:
                # Use thread pool for I/O bound tasks
                future_to_index = {}
                
                for index, item in items_to_process:
                    future = self._thread_pool.submit(processing_func, item)
                    future_to_index[future] = index
                
                # Collect results
                processed_results = {}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        processed_results[index] = result
                        
                        # Cache result
                        if use_cache and self.enable_caching:
                            cache_key = self._generate_cache_key(items[index], processing_func.__name__)
                            ttl = self._calculate_ttl(items[index], result)
                            self.cache.set(cache_key, result, ttl)
                            
                    except Exception as e:
                        logger.error(f"Batch processing error for item {index}: {e}")
                        processed_results[index] = None
                
                # Merge with cached results
                all_results = {**cached_results, **processed_results}
            else:
                # Sequential processing fallback
                all_results = {}
                for index, item in items_to_process:
                    try:
                        result = processing_func(item)
                        all_results[index] = result
                    except Exception as e:
                        logger.error(f"Sequential processing error for item {index}: {e}")
                        all_results[index] = None
                
                # Add cached results
                all_results.update(cached_results)
        else:
            all_results = cached_results
        
        # Build final results list in order
        results = [all_results.get(i) for i in range(len(items))]
        
        # Record batch timing
        batch_time = time.time() - start_time
        self._record_request_time(batch_time / len(items) if items else 0)  # Average per item
        
        return results
    
    def _process_parallel(self, content: str, processing_func: Callable) -> Any:
        """Process single item with parallel optimization."""
        if self._thread_pool:
            future = self._thread_pool.submit(processing_func, content)
            return future.result(timeout=30.0)  # 30 second timeout
        else:
            return processing_func(content)
    
    def _generate_cache_key(self, content: str, func_name: str) -> str:
        """Generate cache key for content and function."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        return f"{func_name}:{content_hash[:16]}"
    
    def _calculate_ttl(self, content: str, result: Any) -> float:
        """Calculate adaptive TTL based on content and result characteristics."""
        base_ttl = 3600.0  # 1 hour base
        
        # Longer TTL for longer content (more processing intensive)
        length_factor = min(len(content) / 1000.0, 2.0)
        
        # Shorter TTL for dynamic content
        dynamic_indicators = ['today', 'now', 'current', 'latest']
        dynamic_factor = 0.5 if any(word in content.lower() for word in dynamic_indicators) else 1.0
        
        return base_ttl * length_factor * dynamic_factor
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on system resources."""
        # Base batch size on available workers and system capacity
        available_workers = self.autoscaler.current_workers
        
        # Optimal batch size balances parallelism and overhead
        if total_items <= available_workers:
            return 1  # One item per worker
        
        # Calculate based on worker capacity and memory considerations
        optimal_batch = max(1, total_items // (available_workers * 2))
        return min(optimal_batch, 50)  # Cap at 50 for memory efficiency
    
    def _record_request_time(self, duration: float, error: bool = False):
        """Record request timing for metrics."""
        with self._lock:
            self._request_times.append(duration)
            
            # Keep only recent times (last 1000)
            if len(self._request_times) > 1000:
                self._request_times = self._request_times[-1000:]
            
            # Update metrics
            self._update_metrics(error)
    
    def _update_metrics(self, error_occurred: bool = False):
        """Update performance metrics."""
        if not self._request_times:
            return
        
        # Calculate timing metrics
        times_ms = [t * 1000 for t in self._request_times[-100:]]  # Last 100 requests
        
        self.metrics.average_response_time_ms = statistics.mean(times_ms)
        
        if len(times_ms) > 1:
            sorted_times = sorted(times_ms)
            p95_idx = int(0.95 * len(sorted_times))
            p99_idx = int(0.99 * len(sorted_times))
            
            self.metrics.p95_response_time_ms = sorted_times[p95_idx]
            self.metrics.p99_response_time_ms = sorted_times[p99_idx]
        
        # Calculate throughput
        recent_times = [t for t in self._request_times if t > time.time() - 60]  # Last minute
        self.metrics.requests_per_second = len(recent_times) / 60.0
        self.metrics.throughput_items_per_minute = len(recent_times)
        
        # Update worker count
        self.metrics.active_workers = self.autoscaler.current_workers
        
        # Cache hit rate
        cache_stats = self.cache.get_statistics()
        self.metrics.cache_hit_rate = cache_stats['hit_rate']
        
        # Error rate (simple approximation)
        if error_occurred:
            self.metrics.error_rate = min(self.metrics.error_rate + 0.01, 1.0)
        else:
            self.metrics.error_rate = max(self.metrics.error_rate - 0.001, 0.0)
    
    def _performance_monitor_loop(self):
        """Performance monitoring and auto-scaling loop."""
        logger.info("Starting performance monitoring loop")
        
        while self._monitoring_active:
            try:
                time.sleep(10)  # Monitor every 10 seconds
                
                # Update metrics
                self._update_metrics()
                
                # Record metrics history
                with self._lock:
                    timestamp = time.time()
                    self._metrics_history.append((timestamp, self.metrics))
                    
                    # Keep only recent history
                    if len(self._metrics_history) > 1000:
                        self._metrics_history = self._metrics_history[-1000:]
                
                # Check if scaling is needed
                scaling_decision = self.autoscaler.should_scale(self.metrics)
                
                if scaling_decision.action != "maintain":
                    # Execute scaling
                    if self.autoscaler.execute_scaling(scaling_decision):
                        # Update thread pool size
                        self._resize_thread_pool(scaling_decision.target_workers)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def _resize_thread_pool(self, target_workers: int):
        """Resize thread pool to match target workers."""
        if self._thread_pool and target_workers != self.autoscaler.current_workers:
            logger.info(f"Resizing thread pool to {target_workers} workers")
            
            # Create new thread pool (ThreadPoolExecutor doesn't support dynamic resizing)
            old_pool = self._thread_pool
            self._thread_pool = ThreadPoolExecutor(
                max_workers=target_workers,
                thread_name_prefix="hyperscale-resized"
            )
            
            # Schedule shutdown of old pool
            def shutdown_old_pool():
                time.sleep(5)  # Give current tasks time to complete
                old_pool.shutdown(wait=True)
            
            threading.Thread(target=shutdown_old_pool, daemon=True).start()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self._lock:
            cache_stats = self.cache.get_statistics()
            
            return {
                'current_metrics': {
                    'requests_per_second': self.metrics.requests_per_second,
                    'average_response_time_ms': self.metrics.average_response_time_ms,
                    'p95_response_time_ms': self.metrics.p95_response_time_ms,
                    'p99_response_time_ms': self.metrics.p99_response_time_ms,
                    'throughput_items_per_minute': self.metrics.throughput_items_per_minute,
                    'active_workers': self.metrics.active_workers,
                    'error_rate': self.metrics.error_rate,
                },
                'cache_performance': cache_stats,
                'scaling_info': {
                    'current_workers': self.autoscaler.current_workers,
                    'min_workers': self.autoscaler.min_workers,
                    'max_workers': self.autoscaler.max_workers,
                    'strategy': self.autoscaler.strategy.value,
                    'recent_scaling_actions': len(self.autoscaler.scaling_history)
                },
                'system_status': {
                    'monitoring_active': self._monitoring_active,
                    'thread_pool_active': self._thread_pool is not None,
                    'process_pool_active': self._process_pool is not None,
                    'caching_enabled': self.enable_caching,
                    'auto_scaling_enabled': self.enable_auto_scaling
                }
            }


# Global performance engine instance
_performance_engine = None

def get_performance_engine() -> HyperScalePerformanceEngine:
    """Get global hyperscale performance engine instance."""
    global _performance_engine
    if _performance_engine is None:
        _performance_engine = HyperScalePerformanceEngine()
    return _performance_engine


def hyperscale_optimization(use_cache: bool = True, use_parallel: bool = True):
    """Decorator for hyperscale performance optimization."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            engine = get_performance_engine()
            
            # For functions that take content as first argument
            if args and isinstance(args[0], str):
                return engine.process_email_optimized(
                    args[0], 
                    lambda content: func(content, *args[1:], **kwargs),
                    use_cache=use_cache,
                    use_parallel=use_parallel
                )
            else:
                # Fallback to regular execution
                return func(*args, **kwargs)
        return wrapper
    return decorator
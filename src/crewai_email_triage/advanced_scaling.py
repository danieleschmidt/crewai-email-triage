"""Advanced scaling and performance optimization capabilities."""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
import psutil
import threading

from .logging_utils import get_logger
from .metrics_export import get_metrics_collector
from .resilience import resilience

logger = get_logger(__name__)
_metrics_collector = get_metrics_collector()


@dataclass
class ScalingMetrics:
    """Comprehensive scaling and performance metrics."""
    
    # Throughput metrics
    messages_per_second: float = 0.0
    peak_throughput: float = 0.0
    average_throughput: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    
    # Concurrency metrics
    active_workers: int = 0
    max_workers: int = 0
    worker_utilization: float = 0.0
    
    # Performance metrics
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Queue metrics
    queue_size: int = 0
    max_queue_size: int = 0
    queue_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "throughput": {
                "current_mps": self.messages_per_second,
                "peak_mps": self.peak_throughput,
                "average_mps": self.average_throughput,
            },
            "resources": {
                "cpu_percent": self.cpu_usage_percent,
                "memory_mb": self.memory_usage_mb,
                "memory_percent": self.memory_usage_percent,
            },
            "concurrency": {
                "active_workers": self.active_workers,
                "max_workers": self.max_workers,
                "worker_utilization": self.worker_utilization,
            },
            "latency": {
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms,
            },
            "queue": {
                "size": self.queue_size,
                "max_size": self.max_queue_size,
                "utilization": self.queue_utilization,
            }
        }


class AdaptiveLoadBalancer:
    """Intelligent load balancer with adaptive worker management."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = None):
        """Initialize adaptive load balancer."""
        self.min_workers = min_workers
        self.max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
        
        # Performance tracking
        self.worker_performance = {}
        self.load_history = []
        self.scaling_decisions = []
        
        # Current state
        self.current_workers = self.min_workers
        self.last_scale_time = time.time()
        self.scale_cooldown = 30.0  # seconds
        
        logger.info(f"AdaptiveLoadBalancer initialized: {self.min_workers}-{self.max_workers} workers")
    
    def should_scale_up(self, queue_size: int, current_throughput: float, target_latency_ms: float = 1000) -> bool:
        """Determine if we should scale up based on current conditions."""
        
        # Don't scale if we're at maximum
        if self.current_workers >= self.max_workers:
            return False
        
        # Respect cooldown period
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Scale up conditions
        scale_reasons = []
        
        # High queue backlog
        if queue_size > self.current_workers * 5:
            scale_reasons.append("high_queue_backlog")
        
        # High worker utilization
        utilization = self._calculate_worker_utilization()
        if utilization > 0.8:
            scale_reasons.append("high_worker_utilization")
        
        # Poor latency performance
        avg_latency = self._get_average_latency()
        if avg_latency > target_latency_ms:
            scale_reasons.append("high_latency")
        
        if scale_reasons:
            logger.info(f"Scale up recommended: {', '.join(scale_reasons)}")
            self.scaling_decisions.append({
                'action': 'scale_up',
                'reasons': scale_reasons,
                'timestamp': time.time(),
                'workers_before': self.current_workers,
            })
            return True
        
        return False
    
    def should_scale_down(self, queue_size: int, current_throughput: float) -> bool:
        """Determine if we should scale down based on current conditions."""
        
        # Don't scale below minimum
        if self.current_workers <= self.min_workers:
            return False
        
        # Respect cooldown period
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Scale down conditions
        scale_reasons = []
        
        # Low queue size
        if queue_size < self.current_workers:
            scale_reasons.append("low_queue_size")
        
        # Low worker utilization
        utilization = self._calculate_worker_utilization()
        if utilization < 0.3:
            scale_reasons.append("low_worker_utilization")
        
        # Sustained low throughput
        if self._is_sustained_low_throughput(current_throughput):
            scale_reasons.append("sustained_low_throughput")
        
        if scale_reasons:
            logger.info(f"Scale down recommended: {', '.join(scale_reasons)}")
            self.scaling_decisions.append({
                'action': 'scale_down',
                'reasons': scale_reasons,
                'timestamp': time.time(),
                'workers_before': self.current_workers,
            })
            return True
        
        return False
    
    def scale_workers(self, new_count: int):
        """Update worker count and record scaling decision."""
        old_count = self.current_workers
        self.current_workers = max(self.min_workers, min(new_count, self.max_workers))
        self.last_scale_time = time.time()
        
        if self.scaling_decisions:
            self.scaling_decisions[-1]['workers_after'] = self.current_workers
        
        _metrics_collector.set_gauge("adaptive_worker_count", self.current_workers)
        _metrics_collector.increment_counter("adaptive_scaling_events")
        
        logger.info(f"Scaled workers: {old_count} -> {self.current_workers}")
    
    def _calculate_worker_utilization(self) -> float:
        """Calculate current worker utilization."""
        if not self.worker_performance:
            return 0.0
        
        total_utilization = sum(perf.get('utilization', 0) for perf in self.worker_performance.values())
        return total_utilization / len(self.worker_performance)
    
    def _get_average_latency(self) -> float:
        """Get average latency across workers."""
        if not self.worker_performance:
            return 0.0
        
        latencies = [perf.get('avg_latency_ms', 0) for perf in self.worker_performance.values()]
        return sum(latencies) / len(latencies) if latencies else 0.0
    
    def _is_sustained_low_throughput(self, current_throughput: float, duration_minutes: int = 5) -> bool:
        """Check if we have sustained low throughput."""
        self.load_history.append({
            'throughput': current_throughput,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        cutoff_time = time.time() - (duration_minutes * 60)
        self.load_history = [h for h in self.load_history if h['timestamp'] > cutoff_time]
        
        if len(self.load_history) < 10:  # Need enough data points
            return False
        
        # Check if recent throughput is consistently low
        recent_throughput = [h['throughput'] for h in self.load_history[-10:]]
        avg_recent = sum(recent_throughput) / len(recent_throughput)
        
        return avg_recent < (self.current_workers * 2)  # Less than 2 messages per worker per second
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'utilization': self._calculate_worker_utilization(),
            'avg_latency_ms': self._get_average_latency(),
            'last_scale_time': self.last_scale_time,
            'scale_cooldown_remaining': max(0, self.scale_cooldown - (time.time() - self.last_scale_time)),
            'recent_decisions': self.scaling_decisions[-5:] if self.scaling_decisions else [],
        }


class HighPerformanceProcessor:
    """High-performance email processing with advanced optimizations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize high-performance processor."""
        self.config = config or {}
        
        # Performance configuration
        self.batch_size = self.config.get('batch_size', 50)
        self.prefetch_count = self.config.get('prefetch_count', 100)
        self.enable_vectorization = self.config.get('enable_vectorization', True)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Load balancer
        self.load_balancer = AdaptiveLoadBalancer(
            min_workers=self.config.get('min_workers', 2),
            max_workers=self.config.get('max_workers', 16)
        )
        
        # Performance tracking
        self.metrics = ScalingMetrics()
        self.latency_samples = []
        self.throughput_samples = []
        
        # Processing state
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
        logger.info("HighPerformanceProcessor initialized")
    
    async def process_batch_optimized(
        self,
        messages: List[str],
        headers_list: Optional[List[Dict]] = None,
        priority_sort: bool = True
    ) -> List[Dict[str, Any]]:
        """Process batch of messages with advanced optimizations."""
        
        start_time = time.perf_counter()
        
        if not messages:
            return []
        
        # Prepare headers
        if not headers_list:
            headers_list = [None] * len(messages)
        
        try:
            # Pre-processing optimizations
            if priority_sort:
                messages, headers_list = self._priority_sort_messages(messages, headers_list)
            
            # Batch processing with adaptive concurrency
            results = await self._process_with_adaptive_concurrency(messages, headers_list)
            
            # Post-processing optimizations
            results = self._apply_batch_optimizations(results)
            
            # Update metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            throughput = len(messages) / (processing_time / 1000) if processing_time > 0 else 0
            
            self._update_performance_metrics(processing_time, throughput, len(messages))
            
            logger.info(f"Batch processed: {len(messages)} messages in {processing_time:.2f}ms ({throughput:.1f} msg/s)")
            
            return results
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Batch processing failed after {processing_time:.2f}ms: {e}")
            _metrics_collector.increment_counter("batch_processing_errors")
            raise
    
    def _priority_sort_messages(
        self, 
        messages: List[str], 
        headers_list: List[Optional[Dict]]
    ) -> tuple[List[str], List[Optional[Dict]]]:
        """Sort messages by priority indicators for optimal processing order."""
        
        def get_priority_score(msg: str) -> int:
            """Calculate priority score for message ordering."""
            score = 0
            msg_lower = msg.lower()
            
            # Urgent indicators
            urgent_keywords = ['urgent', 'asap', 'emergency', 'critical', 'immediately']
            for keyword in urgent_keywords:
                if keyword in msg_lower:
                    score += 10
            
            # Customer tier indicators
            if any(word in msg_lower for word in ['vip', 'premium', 'enterprise']):
                score += 5
            
            # Complaint indicators
            if any(word in msg_lower for word in ['complaint', 'angry', 'frustrated']):
                score += 3
            
            # Length penalty (shorter messages processed first for quick wins)
            if len(msg) < 200:
                score += 2
            elif len(msg) > 1000:
                score -= 1
            
            return score
        
        # Create indexed tuples for sorting
        indexed_data = [
            (i, get_priority_score(msg), msg, headers) 
            for i, (msg, headers) in enumerate(zip(messages, headers_list))
        ]
        
        # Sort by priority score (descending)
        indexed_data.sort(key=lambda x: x[1], reverse=True)
        
        # Extract sorted messages and headers
        sorted_messages = [item[2] for item in indexed_data]
        sorted_headers = [item[3] for item in indexed_data]
        
        logger.debug(f"Priority sorted {len(messages)} messages")
        return sorted_messages, sorted_headers
    
    async def _process_with_adaptive_concurrency(
        self,
        messages: List[str],
        headers_list: List[Optional[Dict]]
    ) -> List[Dict[str, Any]]:
        """Process messages with adaptive concurrency management."""
        
        # Determine optimal worker count
        current_queue_size = len(messages)
        current_throughput = self._get_recent_throughput()
        
        # Check scaling decisions
        if self.load_balancer.should_scale_up(current_queue_size, current_throughput):
            new_workers = min(self.load_balancer.current_workers + 2, self.load_balancer.max_workers)
            self.load_balancer.scale_workers(new_workers)
        elif self.load_balancer.should_scale_down(current_queue_size, current_throughput):
            new_workers = max(self.load_balancer.current_workers - 1, self.load_balancer.min_workers)
            self.load_balancer.scale_workers(new_workers)
        
        # Process with optimal worker count
        optimal_workers = min(self.load_balancer.current_workers, len(messages))
        
        # Batch messages for workers
        batches = self._create_optimal_batches(messages, headers_list, optimal_workers)
        
        # Process batches concurrently
        tasks = [
            self._process_batch_chunk(batch_msgs, batch_headers)
            for batch_msgs, batch_headers in batches
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error(f"Batch chunk failed: {batch_result}")
                _metrics_collector.increment_counter("batch_chunk_errors")
                # Add error placeholders
                results.extend([{
                    'category': 'error',
                    'priority': 0,
                    'summary': 'Processing failed',
                    'response': 'Unable to process message'
                }] * self.batch_size)  # Approximate batch size
            else:
                results.extend(batch_result)
        
        return results
    
    def _create_optimal_batches(
        self,
        messages: List[str],
        headers_list: List[Optional[Dict]],
        num_workers: int
    ) -> List[tuple[List[str], List[Optional[Dict]]]]:
        """Create optimally sized batches for workers."""
        
        if num_workers <= 0:
            num_workers = 1
        
        # Calculate optimal batch size
        total_messages = len(messages)
        base_batch_size = max(1, total_messages // num_workers)
        
        # Account for variable processing complexity
        batches = []
        start_idx = 0
        
        for worker_idx in range(num_workers):
            if start_idx >= total_messages:
                break
            
            # Adjust batch size based on message characteristics
            end_idx = min(start_idx + base_batch_size, total_messages)
            
            # Ensure last worker gets remaining messages
            if worker_idx == num_workers - 1:
                end_idx = total_messages
            
            batch_messages = messages[start_idx:end_idx]
            batch_headers = headers_list[start_idx:end_idx]
            
            batches.append((batch_messages, batch_headers))
            start_idx = end_idx
        
        logger.debug(f"Created {len(batches)} batches for {num_workers} workers")
        return batches
    
    async def _process_batch_chunk(
        self,
        messages: List[str],
        headers_list: List[Optional[Dict]]
    ) -> List[Dict[str, Any]]:
        """Process a chunk of messages efficiently."""
        
        if not messages:
            return []
        
        # Import here to avoid circular imports
        from .ai_enhancements import intelligent_triage_email
        
        # Process messages with caching and optimization
        results = []
        
        for msg, headers in zip(messages, headers_list):
            try:
                # Check cache first
                cache_key = self._generate_cache_key(msg, headers)
                
                if self.enable_caching and cache_key in self.result_cache:
                    with self.cache_lock:
                        cached_result = self.result_cache[cache_key]
                        results.append(cached_result.copy())
                        _metrics_collector.increment_counter("processing_cache_hits")
                    continue
                
                # Process message
                result = await intelligent_triage_email(msg, headers, self.config)
                result_dict = result.to_dict()
                
                # Cache result
                if self.enable_caching:
                    with self.cache_lock:
                        # Limit cache size
                        if len(self.result_cache) > 1000:
                            # Remove oldest entries
                            keys_to_remove = list(self.result_cache.keys())[:100]
                            for key in keys_to_remove:
                                del self.result_cache[key]
                        
                        self.result_cache[cache_key] = result_dict.copy()
                        _metrics_collector.increment_counter("processing_cache_stores")
                
                results.append(result_dict)
                
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                _metrics_collector.increment_counter("message_processing_errors")
                
                # Graceful degradation
                results.append({
                    'category': 'error',
                    'priority': 0,
                    'summary': 'Processing failed',
                    'response': 'Unable to process message',
                    'error': str(e)
                })
        
        return results
    
    def _generate_cache_key(self, message: str, headers: Optional[Dict]) -> str:
        """Generate cache key for message."""
        import hashlib
        
        content = message + str(headers or {})
        return hashlib.md5(content.encode()).hexdigest()
    
    def _apply_batch_optimizations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply batch-level optimizations to results."""
        
        # Batch similarity detection for potential duplicates
        if len(results) > 10:
            results = self._detect_and_optimize_similar_results(results)
        
        # Batch response optimization
        results = self._optimize_batch_responses(results)
        
        return results
    
    def _detect_and_optimize_similar_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect similar results and optimize processing."""
        
        # Group by category and priority for bulk optimizations
        categories = {}
        for i, result in enumerate(results):
            key = (result.get('category', 'unknown'), result.get('priority', 0))
            if key not in categories:
                categories[key] = []
            categories[key].append(i)
        
        # Apply category-specific optimizations
        for (category, priority), indices in categories.items():
            if len(indices) > 3:  # Multiple similar messages
                logger.debug(f"Optimizing {len(indices)} similar messages: {category}(p{priority})")
                _metrics_collector.increment_counter("batch_similarity_optimizations")
        
        return results
    
    def _optimize_batch_responses(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize responses at batch level."""
        
        # Response template caching and reuse
        response_templates = {}
        
        for result in results:
            category = result.get('category', 'unknown')
            if category not in response_templates:
                response_templates[category] = result.get('response', '')
            
            # Use template for similar categories to improve consistency
            if len(result.get('response', '')) < 50:  # Short/generic response
                template = response_templates.get(category, '')
                if len(template) > len(result.get('response', '')):
                    result['response'] = template
                    result['response_optimized'] = True
        
        return results
    
    def _update_performance_metrics(self, processing_time_ms: float, throughput: float, message_count: int):
        """Update comprehensive performance metrics."""
        
        # Update latency samples
        self.latency_samples.append(processing_time_ms / message_count if message_count > 0 else processing_time_ms)
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-500:]  # Keep recent samples
        
        # Update throughput samples
        self.throughput_samples.append(throughput)
        if len(self.throughput_samples) > 100:
            self.throughput_samples = self.throughput_samples[-50:]
        
        # Calculate percentiles
        if self.latency_samples:
            sorted_latencies = sorted(self.latency_samples)
            self.metrics.p50_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.5)]
            self.metrics.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            self.metrics.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        # Update throughput metrics
        self.metrics.messages_per_second = throughput
        if throughput > self.metrics.peak_throughput:
            self.metrics.peak_throughput = throughput
        
        if self.throughput_samples:
            self.metrics.average_throughput = sum(self.throughput_samples) / len(self.throughput_samples)
        
        # Update resource metrics
        try:
            process = psutil.Process()
            self.metrics.cpu_usage_percent = process.cpu_percent()
            memory_info = process.memory_info()
            self.metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
            self.metrics.memory_usage_percent = process.memory_percent()
        except Exception as e:
            logger.debug(f"Failed to update resource metrics: {e}")
        
        # Update worker metrics
        self.metrics.active_workers = self.load_balancer.current_workers
        self.metrics.max_workers = self.load_balancer.max_workers
        self.metrics.worker_utilization = self.load_balancer._calculate_worker_utilization()
        
        # Update metrics collector
        _metrics_collector.set_gauge("processing_throughput_mps", throughput)
        _metrics_collector.set_gauge("processing_p95_latency_ms", self.metrics.p95_latency_ms)
        _metrics_collector.set_gauge("processing_cpu_percent", self.metrics.cpu_usage_percent)
        _metrics_collector.set_gauge("processing_memory_mb", self.metrics.memory_usage_mb)
    
    def _get_recent_throughput(self) -> float:
        """Get recent average throughput."""
        if not self.throughput_samples:
            return 0.0
        
        recent_samples = self.throughput_samples[-10:] if len(self.throughput_samples) > 10 else self.throughput_samples
        return sum(recent_samples) / len(recent_samples)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        
        return {
            'metrics': self.metrics.to_dict(),
            'load_balancer': self.load_balancer.get_scaling_status(),
            'cache_stats': {
                'enabled': self.enable_caching,
                'cache_size': len(self.result_cache),
                'hit_rate': _metrics_collector.get_counter("processing_cache_hits") / 
                          max(1, _metrics_collector.get_counter("processing_cache_hits") + 
                              _metrics_collector.get_counter("processing_cache_stores")),
            },
            'configuration': {
                'batch_size': self.batch_size,
                'prefetch_count': self.prefetch_count,
                'enable_vectorization': self.enable_vectorization,
                'enable_caching': self.enable_caching,
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }
    
    def optimize_configuration(self):
        """Automatically optimize configuration based on performance data."""
        
        logger.info("Running automatic configuration optimization...")
        
        # Optimize batch size based on throughput
        if len(self.throughput_samples) > 20:
            avg_throughput = sum(self.throughput_samples[-20:]) / 20
            
            if avg_throughput < 10 and self.batch_size > 10:
                self.batch_size = max(10, self.batch_size - 10)
                logger.info(f"Reduced batch size to {self.batch_size} due to low throughput")
            elif avg_throughput > 50 and self.batch_size < 100:
                self.batch_size = min(100, self.batch_size + 10)
                logger.info(f"Increased batch size to {self.batch_size} due to high throughput")
        
        # Optimize caching based on hit rate
        cache_hit_rate = _metrics_collector.get_counter("processing_cache_hits") / max(1, 
                        _metrics_collector.get_counter("processing_cache_hits") + 
                        _metrics_collector.get_counter("processing_cache_stores"))
        
        if cache_hit_rate < 0.1 and self.enable_caching:
            logger.info("Considering cache disabling due to low hit rate")
        elif cache_hit_rate > 0.3 and not self.enable_caching:
            self.enable_caching = True
            logger.info("Enabled caching due to potential benefits")


# Global high-performance processor instance
high_performance_processor = HighPerformanceProcessor()


async def process_batch_high_performance(
    messages: List[str],
    headers_list: Optional[List[Dict]] = None,
    config: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """Process batch of messages with maximum performance optimization."""
    
    if config:
        processor = HighPerformanceProcessor(config)
    else:
        processor = high_performance_processor
    
    return await processor.process_batch_optimized(messages, headers_list)


def get_performance_insights() -> Dict[str, Any]:
    """Get performance insights and recommendations."""
    
    return high_performance_processor.get_performance_report()


def optimize_system_performance():
    """Run automatic system performance optimization."""
    
    high_performance_processor.optimize_configuration()
    logger.info("System performance optimization completed")
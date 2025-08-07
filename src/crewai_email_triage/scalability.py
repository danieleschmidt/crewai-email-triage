"""Advanced scalability and performance optimization features."""

from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .cache import get_smart_cache
from .logging_utils import get_logger
from .metrics_export import get_metrics_collector
from .pipeline import TriageResult, triage_batch, triage_email_enhanced

logger = get_logger(__name__)
_metrics_collector = get_metrics_collector()


@dataclass
class BatchOptimizationConfig:
    """Configuration for batch processing optimization."""
    
    # Parallelization settings
    adaptive_workers: bool = True
    min_workers: int = 1
    max_workers: int = 16
    worker_scaling_factor: float = 0.5
    
    # Batching settings
    optimal_batch_size: int = 100
    max_batch_size: int = 1000
    min_batch_size: int = 10
    
    # Caching settings
    enable_intelligent_caching: bool = True
    cache_similarity_threshold: float = 0.85
    cache_ttl_seconds: int = 3600
    
    # Performance settings
    timeout_seconds: int = 300
    retry_failed_items: bool = True
    max_retries: int = 3


@dataclass 
class ProcessingStats:
    """Statistics and metrics for batch processing operations."""
    
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    cached_items: int = 0
    
    processing_time_seconds: float = 0.0
    throughput_items_per_second: float = 0.0
    
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    
    worker_utilization: float = 0.0
    optimal_workers_used: int = 0
    
    performance_grade: str = "A"
    recommendations: List[str] = field(default_factory=list)


class AdaptiveScalingProcessor:
    """High-performance processor with adaptive scaling and intelligent optimization."""
    
    def __init__(self, config: Optional[BatchOptimizationConfig] = None):
        self.config = config or BatchOptimizationConfig()
        self.cache = get_smart_cache()
        self._processing_stats = ProcessingStats()
        
    def process_batch_optimized(
        self, 
        messages: List[str], 
        config_dict: Optional[Dict] = None
    ) -> tuple[List[Dict[str, Any]], ProcessingStats]:
        """Process batch with adaptive scaling and intelligent optimization."""
        start_time = time.perf_counter()
        
        logger.info(f"Starting optimized batch processing",
                   extra={'message_count': len(messages),
                         'adaptive_workers': self.config.adaptive_workers})
        
        # Initialize stats
        stats = ProcessingStats(total_items=len(messages))
        
        # Determine optimal worker count
        optimal_workers = self._calculate_optimal_workers(len(messages))
        stats.optimal_workers_used = optimal_workers
        
        # Process with intelligent batching and caching
        try:
            results = self._process_with_adaptive_scaling(
                messages, config_dict, optimal_workers, stats
            )
            
            # Calculate final statistics
            elapsed = time.perf_counter() - start_time
            stats.processing_time_seconds = elapsed
            stats.throughput_items_per_second = len(messages) / elapsed if elapsed > 0 else 0
            stats.cache_hit_rate = stats.cached_items / len(messages) if len(messages) > 0 else 0
            stats.error_rate = stats.failed_items / len(messages) if len(messages) > 0 else 0
            
            # Generate performance recommendations
            stats.recommendations = self._generate_recommendations(stats)
            stats.performance_grade = self._calculate_performance_grade(stats)
            
            # Update metrics
            _metrics_collector.record_histogram("adaptive_batch_processing_time", elapsed)
            _metrics_collector.set_gauge("adaptive_batch_throughput", stats.throughput_items_per_second)
            _metrics_collector.set_gauge("adaptive_cache_hit_rate", stats.cache_hit_rate)
            
            logger.info(f"Batch processing completed",
                       extra={
                           'throughput': f"{stats.throughput_items_per_second:.1f} items/sec",
                           'cache_hit_rate': f"{stats.cache_hit_rate:.1%}",
                           'performance_grade': stats.performance_grade,
                           'workers_used': optimal_workers
                       })
            
            return results, stats
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
            stats.error_rate = 1.0
            stats.performance_grade = "F"
            stats.recommendations.append("Fix critical processing errors")
            return [], stats
            
    def _calculate_optimal_workers(self, message_count: int) -> int:
        """Calculate optimal number of workers based on message count and system resources."""
        if not self.config.adaptive_workers:
            return self.config.min_workers
            
        # Base calculation on message count and processing complexity
        suggested_workers = int(message_count * self.config.worker_scaling_factor)
        
        # Clamp to configured bounds
        optimal_workers = max(
            self.config.min_workers,
            min(suggested_workers, self.config.max_workers)
        )
        
        logger.debug(f"Calculated optimal workers: {optimal_workers} for {message_count} messages")
        return optimal_workers
        
    def _process_with_adaptive_scaling(
        self, 
        messages: List[str], 
        config_dict: Optional[Dict],
        workers: int,
        stats: ProcessingStats
    ) -> List[Dict[str, Any]]:
        """Process messages with adaptive scaling and intelligent caching."""
        
        # Check cache for similar messages
        cache_results = {}
        remaining_messages = []
        message_indices = {}
        
        if self.config.enable_intelligent_caching:
            for i, message in enumerate(messages):
                cache_key = self._generate_cache_key(message)
                cached_result = self.cache.get_agent_result("batch_processor", message, cache_key)
                
                if cached_result:
                    cache_results[i] = cached_result
                    stats.cached_items += 1
                else:
                    remaining_messages.append(message)
                    message_indices[len(remaining_messages) - 1] = i
        else:
            remaining_messages = messages
            message_indices = {i: i for i in range(len(messages))}
        
        # Process remaining messages
        if remaining_messages:
            if workers > 1 and len(remaining_messages) > self.config.min_batch_size:
                processed_results = self._parallel_process(remaining_messages, config_dict, workers)
            else:
                processed_results = self._sequential_process(remaining_messages, config_dict)
                
            # Cache new results
            if self.config.enable_intelligent_caching:
                for i, (message, result) in enumerate(zip(remaining_messages, processed_results)):
                    cache_key = self._generate_cache_key(message)
                    self.cache.set_agent_result("batch_processor", message, cache_key, result)
        else:
            processed_results = []
        
        # Combine cached and processed results
        final_results = [None] * len(messages)
        
        # Fill in cached results
        for original_idx, result in cache_results.items():
            final_results[original_idx] = result
            
        # Fill in processed results
        for processed_idx, result in enumerate(processed_results):
            original_idx = message_indices[processed_idx]
            final_results[original_idx] = result
            
        stats.processed_items = len([r for r in final_results if r is not None])
        stats.failed_items = len(messages) - stats.processed_items
        
        return final_results
        
    def _parallel_process(
        self, messages: List[str], config_dict: Optional[Dict], workers: int
    ) -> List[Dict[str, Any]]:
        """Process messages in parallel with the specified number of workers."""
        return triage_batch(messages, parallel=True, max_workers=workers, config_dict=config_dict)
        
    def _sequential_process(
        self, messages: List[str], config_dict: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """Process messages sequentially."""
        return triage_batch(messages, parallel=False, config_dict=config_dict)
        
    def _generate_cache_key(self, message: str) -> str:
        """Generate a cache key for message content."""
        # Use content hash for cache key
        return hashlib.sha256(message.encode('utf-8')).hexdigest()[:16]
        
    def _generate_recommendations(self, stats: ProcessingStats) -> List[str]:
        """Generate performance recommendations based on processing stats."""
        recommendations = []
        
        if stats.cache_hit_rate < 0.3:
            recommendations.append("Consider enabling intelligent caching for repeated content")
            
        if stats.throughput_items_per_second < 5.0:
            recommendations.append("Performance is below optimal - consider increasing worker count")
            
        if stats.error_rate > 0.05:
            recommendations.append("High error rate detected - review input validation")
            
        if stats.cache_hit_rate > 0.8:
            recommendations.append("Excellent cache performance - system is well optimized")
            
        if stats.throughput_items_per_second > 20.0:
            recommendations.append("Outstanding throughput performance")
            
        if not recommendations:
            recommendations.append("Performance looks good - no immediate optimizations needed")
            
        return recommendations
        
    def _calculate_performance_grade(self, stats: ProcessingStats) -> str:
        """Calculate overall performance grade."""
        score = 0
        
        # Throughput scoring (40% of grade)
        if stats.throughput_items_per_second > 20:
            score += 40
        elif stats.throughput_items_per_second > 10:
            score += 30
        elif stats.throughput_items_per_second > 5:
            score += 20
        else:
            score += 10
            
        # Error rate scoring (30% of grade) 
        if stats.error_rate < 0.01:
            score += 30
        elif stats.error_rate < 0.05:
            score += 20
        else:
            score += 5
            
        # Cache efficiency scoring (30% of grade)
        if stats.cache_hit_rate > 0.7:
            score += 30
        elif stats.cache_hit_rate > 0.4:
            score += 20
        elif stats.cache_hit_rate > 0.1:
            score += 10
        else:
            score += 5
            
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B" 
        elif score >= 55:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"


# Global adaptive processor instance
_adaptive_processor: Optional[AdaptiveScalingProcessor] = None


def get_adaptive_processor(config: Optional[BatchOptimizationConfig] = None) -> AdaptiveScalingProcessor:
    """Get the global adaptive scaling processor instance."""
    global _adaptive_processor
    if _adaptive_processor is None:
        _adaptive_processor = AdaptiveScalingProcessor(config)
    return _adaptive_processor


def process_batch_with_scaling(
    messages: List[str], 
    config_dict: Optional[Dict] = None
) -> tuple[List[Dict[str, Any]], ProcessingStats]:
    """Process batch with automatic scaling optimization."""
    processor = get_adaptive_processor()
    return processor.process_batch_optimized(messages, config_dict)


def benchmark_performance(
    test_messages: Optional[List[str]] = None,
    iterations: int = 3
) -> Dict[str, Any]:
    """Benchmark system performance across different configurations."""
    
    if test_messages is None:
        test_messages = [
            f"Performance benchmark test message {i} with realistic content for email processing"
            for i in range(50)
        ]
    
    logger.info("Starting performance benchmark", extra={'iterations': iterations})
    
    configs_to_test = [
        ("Sequential", {"parallel": False}),
        ("Parallel-2", {"parallel": True, "max_workers": 2}),
        ("Parallel-4", {"parallel": True, "max_workers": 4}),
        ("Adaptive", {"use_adaptive": True}),
    ]
    
    benchmark_results = {}
    
    for config_name, config_params in configs_to_test:
        times = []
        
        for iteration in range(iterations):
            start_time = time.perf_counter()
            
            if config_params.get("use_adaptive"):
                results, stats = process_batch_with_scaling(test_messages)
                throughput = stats.throughput_items_per_second
            else:
                results = triage_batch(
                    test_messages,
                    parallel=config_params.get("parallel", False),
                    max_workers=config_params.get("max_workers")
                )
                elapsed = time.perf_counter() - start_time
                throughput = len(test_messages) / elapsed if elapsed > 0 else 0
                
            times.append(time.perf_counter() - start_time)
            
        avg_time = sum(times) / len(times)
        avg_throughput = len(test_messages) / avg_time if avg_time > 0 else 0
        
        benchmark_results[config_name] = {
            "avg_time_seconds": avg_time,
            "throughput_items_per_second": avg_throughput,
            "times": times
        }
    
    # Find best performer
    best_config = max(benchmark_results.keys(), 
                     key=lambda k: benchmark_results[k]["throughput_items_per_second"])
    
    benchmark_results["summary"] = {
        "best_configuration": best_config,
        "best_throughput": benchmark_results[best_config]["throughput_items_per_second"],
        "test_message_count": len(test_messages),
        "iterations": iterations
    }
    
    logger.info("Performance benchmark completed", 
               extra={
                   "best_config": best_config,
                   "best_throughput": f"{benchmark_results[best_config]['throughput_items_per_second']:.1f} items/sec"
               })
    
    return benchmark_results
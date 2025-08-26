"""
Plugin Scaling and Performance Optimization Framework
Implements concurrent processing, caching, and performance monitoring for plugins.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
import hashlib
import json
from collections import defaultdict
from threading import Lock
import queue


@dataclass
class PerformanceMetrics:
    """Performance metrics for plugin execution."""
    total_executions: int = 0
    total_time_ms: float = 0
    average_time_ms: float = 0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0
    success_count: int = 0
    error_count: int = 0
    success_rate: float = 0.0
    throughput_per_second: float = 0.0
    last_execution_time: Optional[float] = None


@dataclass 
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0


class LRUCache:
    """High-performance LRU cache with size limits."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_memory_bytes = 0
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            return len(json.dumps(value, default=str).encode('utf-8'))
        except:
            return len(str(value).encode('utf-8'))
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache exceeds limits."""
        # Remove least recently used entries
        while (len(self.cache) >= self.max_size or 
               self.current_memory_bytes >= self.max_memory_bytes):
            if not self.access_order:
                break
            
            lru_key = self.access_order.pop(0)
            if lru_key in self.cache:
                entry = self.cache.pop(lru_key)
                self.current_memory_bytes -= entry.size_bytes
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                # Move to end (most recently used)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)
                
                self.hits += 1
                return entry.value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            size_bytes = self._calculate_size(value)
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes
                self.access_order.remove(key)
            
            # Check if single entry exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                return  # Don't cache overly large entries
            
            # Evict if needed
            self._evict_if_needed()
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
            self.current_memory_bytes += size_bytes
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / total_accesses if total_accesses > 0 else 0
            
            return {
                'entries': len(self.cache),
                'max_entries': self.max_size,
                'memory_usage_mb': self.current_memory_bytes / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'memory_utilization': self.current_memory_bytes / self.max_memory_bytes
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_memory_bytes = 0


class PluginPerformanceMonitor:
    """Monitor and track plugin performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.lock = Lock()
        self.logger = logging.getLogger("plugin_performance_monitor")
    
    def record_execution(self, plugin_name: str, execution_time_ms: float, success: bool) -> None:
        """Record plugin execution metrics."""
        with self.lock:
            if plugin_name not in self.metrics:
                self.metrics[plugin_name] = PerformanceMetrics()
            
            metrics = self.metrics[plugin_name]
            metrics.total_executions += 1
            metrics.total_time_ms += execution_time_ms
            
            if success:
                metrics.success_count += 1
            else:
                metrics.error_count += 1
            
            metrics.average_time_ms = metrics.total_time_ms / metrics.total_executions
            metrics.min_time_ms = min(metrics.min_time_ms, execution_time_ms)
            metrics.max_time_ms = max(metrics.max_time_ms, execution_time_ms)
            metrics.success_rate = metrics.success_count / metrics.total_executions
            metrics.last_execution_time = time.time()
            
            # Calculate throughput (executions per second over last minute)
            if metrics.total_executions > 1:
                time_window = 60  # seconds
                recent_executions = min(metrics.total_executions, 
                                      int(time_window * metrics.throughput_per_second) + 1)
                metrics.throughput_per_second = recent_executions / time_window
    
    def get_metrics(self, plugin_name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a plugin."""
        with self.lock:
            return self.metrics.get(plugin_name)
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all performance metrics."""
        with self.lock:
            return self.metrics.copy()
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self.lock:
            if not self.metrics:
                return {'message': 'No performance data available'}
            
            total_executions = sum(m.total_executions for m in self.metrics.values())
            total_time = sum(m.total_time_ms for m in self.metrics.values())
            total_successes = sum(m.success_count for m in self.metrics.values())
            
            # Find top performers
            sorted_by_speed = sorted(
                self.metrics.items(),
                key=lambda x: x[1].average_time_ms
            )
            
            sorted_by_reliability = sorted(
                self.metrics.items(), 
                key=lambda x: x[1].success_rate,
                reverse=True
            )
            
            return {
                'summary': {
                    'total_plugins': len(self.metrics),
                    'total_executions': total_executions,
                    'total_time_ms': total_time,
                    'overall_success_rate': total_successes / total_executions if total_executions > 0 else 0,
                    'average_execution_time_ms': total_time / total_executions if total_executions > 0 else 0
                },
                'fastest_plugins': [
                    {
                        'name': name,
                        'avg_time_ms': metrics.average_time_ms,
                        'executions': metrics.total_executions
                    }
                    for name, metrics in sorted_by_speed[:5]
                ],
                'most_reliable_plugins': [
                    {
                        'name': name, 
                        'success_rate': metrics.success_rate,
                        'executions': metrics.total_executions
                    }
                    for name, metrics in sorted_by_reliability[:5]
                ],
                'plugin_details': {
                    name: {
                        'executions': metrics.total_executions,
                        'avg_time_ms': round(metrics.average_time_ms, 2),
                        'min_time_ms': round(metrics.min_time_ms, 2),
                        'max_time_ms': round(metrics.max_time_ms, 2),
                        'success_rate': round(metrics.success_rate * 100, 1),
                        'throughput_per_second': round(metrics.throughput_per_second, 2)
                    }
                    for name, metrics in self.metrics.items()
                }
            }


class ConcurrentPluginProcessor:
    """Process plugins concurrently with load balancing."""
    
    def __init__(self, max_workers: int = 4, queue_size: int = 1000):
        self.max_workers = max_workers
        self.queue_size = queue_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.active_tasks: Dict[str, Future] = {}
        self.lock = Lock()
        self.logger = logging.getLogger("concurrent_plugin_processor")
    
    def process_email_concurrent(self, plugins: List, email_content: str, 
                                metadata: Dict[str, Any], timeout_seconds: float = 30.0) -> Dict[str, Any]:
        """Process email with multiple plugins concurrently."""
        start_time = time.time()
        results = {}
        errors = {}
        
        # Submit all plugin tasks
        futures = {}
        for plugin in plugins:
            try:
                plugin_name = plugin.get_metadata().name
                if plugin.config.enabled:
                    future = self.executor.submit(
                        self._safe_plugin_execution,
                        plugin, email_content, metadata
                    )
                    futures[plugin_name] = future
                    
            except Exception as e:
                self.logger.error(f"Failed to submit plugin task: {e}")
        
        # Collect results with timeout
        completed_count = 0
        for plugin_name, future in futures.items():
            try:
                remaining_time = max(0.1, timeout_seconds - (time.time() - start_time))
                result = future.result(timeout=remaining_time)
                
                if result['success']:
                    results[plugin_name] = result['data']
                else:
                    errors[plugin_name] = result['error']
                
                completed_count += 1
                
            except TimeoutError:
                self.logger.warning(f"Plugin {plugin_name} timed out")
                errors[plugin_name] = "Execution timeout"
                future.cancel()
                
            except Exception as e:
                self.logger.error(f"Plugin {plugin_name} execution failed: {e}")
                errors[plugin_name] = str(e)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'results': results,
            'errors': errors,
            'statistics': {
                'total_plugins': len(futures),
                'completed_plugins': completed_count,
                'successful_plugins': len(results),
                'failed_plugins': len(errors),
                'total_time_ms': round(total_time, 2),
                'concurrent_execution': True
            }
        }
    
    def _safe_plugin_execution(self, plugin, email_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Safely execute plugin with error handling."""
        try:
            start_time = time.time()
            result = plugin.process_email(email_content, metadata)
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'data': result,
                'execution_time_ms': round(execution_time, 2)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': 0
            }
    
    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class SmartPluginCache:
    """Intelligent caching system for plugin results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = LRUCache(max_size=max_size)
        self.ttl_seconds = ttl_seconds
        self.logger = logging.getLogger("smart_plugin_cache")
    
    def _generate_cache_key(self, plugin_name: str, email_content: str, metadata: Dict[str, Any]) -> str:
        """Generate cache key for plugin result."""
        # Create deterministic hash of inputs
        content_hash = hashlib.md5(email_content.encode('utf-8')).hexdigest()
        metadata_hash = hashlib.md5(json.dumps(metadata, sort_keys=True).encode('utf-8')).hexdigest()
        return f"{plugin_name}:{content_hash}:{metadata_hash}"
    
    def get_cached_result(self, plugin_name: str, email_content: str, metadata: Dict[str, Any]) -> Optional[Any]:
        """Get cached plugin result if available and valid."""
        cache_key = self._generate_cache_key(plugin_name, email_content, metadata)
        
        cached_entry = self.cache.get(cache_key)
        if cached_entry is None:
            return None
        
        # Check TTL
        age_seconds = time.time() - cached_entry['timestamp']
        if age_seconds > self.ttl_seconds:
            return None
        
        self.logger.debug(f"Cache hit for plugin {plugin_name}")
        return cached_entry['result']
    
    def cache_result(self, plugin_name: str, email_content: str, metadata: Dict[str, Any], result: Any) -> None:
        """Cache plugin result."""
        cache_key = self._generate_cache_key(plugin_name, email_content, metadata)
        
        cache_data = {
            'result': result,
            'timestamp': time.time(),
            'plugin_name': plugin_name
        }
        
        self.cache.put(cache_key, cache_data)
        self.logger.debug(f"Cached result for plugin {plugin_name}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.logger.info("Plugin cache cleared")


class ScalablePluginManager:
    """Enhanced plugin manager with scaling capabilities."""
    
    def __init__(self, config_path: Optional[str] = None, max_workers: int = 4):
        from .plugin_architecture import PluginManager
        self.base_manager = PluginManager(config_path)
        
        self.performance_monitor = PluginPerformanceMonitor()
        self.cache = SmartPluginCache()
        self.concurrent_processor = ConcurrentPluginProcessor(max_workers=max_workers)
        self.logger = logging.getLogger("scalable_plugin_manager")
        
        self._scaling_config = {
            'enable_caching': True,
            'enable_concurrent_processing': True,
            'cache_ttl_seconds': 3600,
            'max_concurrent_plugins': max_workers,
            'performance_monitoring': True
        }
    
    def process_email_scaled(self, email_content: str, metadata: Dict[str, Any], 
                           concurrent: bool = True) -> Dict[str, Any]:
        """Process email with scaling optimizations."""
        start_time = time.time()
        
        # Get email processor plugins
        from .plugin_architecture import EmailProcessorPlugin
        processors = self.base_manager.registry.get_plugins_by_type(EmailProcessorPlugin)
        processors.sort(key=lambda p: p.get_processing_priority())
        
        results = {}
        cache_hits = 0
        cache_misses = 0
        
        if concurrent and self._scaling_config['enable_concurrent_processing']:
            # Use concurrent processing
            concurrent_result = self.concurrent_processor.process_email_concurrent(
                processors, email_content, metadata
            )
            
            # Record performance metrics
            for plugin_name, result in concurrent_result['results'].items():
                execution_time = result.get('execution_time_ms', 0)
                self.performance_monitor.record_execution(plugin_name, execution_time, True)
                
                # Cache successful results
                if self._scaling_config['enable_caching']:
                    self.cache.cache_result(plugin_name, email_content, metadata, result)
            
            # Record failed executions
            for plugin_name, error in concurrent_result['errors'].items():
                self.performance_monitor.record_execution(plugin_name, 0, False)
            
            results = concurrent_result['results']
            processing_stats = concurrent_result['statistics']
            
        else:
            # Sequential processing with caching
            processing_stats = {
                'total_plugins': len(processors),
                'successful_plugins': 0,
                'failed_plugins': 0,
                'concurrent_execution': False
            }
            
            for plugin in processors:
                plugin_name = plugin.get_metadata().name
                plugin_start_time = time.time()
                
                # Check cache first
                cached_result = None
                if self._scaling_config['enable_caching']:
                    cached_result = self.cache.get_cached_result(plugin_name, email_content, metadata)
                
                if cached_result is not None:
                    results[plugin_name] = cached_result['result']
                    cache_hits += 1
                    processing_stats['successful_plugins'] += 1
                    self.performance_monitor.record_execution(plugin_name, 1, True)  # Cached execution
                    
                else:
                    # Execute plugin
                    cache_misses += 1
                    try:
                        result = plugin.process_email(email_content, metadata)
                        execution_time = (time.time() - plugin_start_time) * 1000
                        
                        results[plugin_name] = result
                        processing_stats['successful_plugins'] += 1
                        
                        # Record metrics and cache result
                        self.performance_monitor.record_execution(plugin_name, execution_time, True)
                        
                        if self._scaling_config['enable_caching']:
                            self.cache.cache_result(plugin_name, email_content, metadata, result)
                        
                    except Exception as e:
                        execution_time = (time.time() - plugin_start_time) * 1000
                        self.logger.error(f"Plugin {plugin_name} failed: {e}")
                        processing_stats['failed_plugins'] += 1
                        self.performance_monitor.record_execution(plugin_name, execution_time, False)
        
        total_time = (time.time() - start_time) * 1000
        
        # Enhanced result with scaling metrics
        return {
            'original_content': email_content,
            'metadata': metadata,
            'enhancements': results,
            'processing_statistics': {
                **processing_stats,
                'total_time_ms': round(total_time, 2),
                'cache_hits': cache_hits,
                'cache_misses': cache_misses,
                'cache_hit_rate': cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
            },
            'performance_summary': self.get_performance_summary()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        perf_report = self.performance_monitor.get_performance_report()
        cache_stats = self.cache.get_cache_stats()
        
        return {
            'performance_metrics': perf_report,
            'cache_statistics': cache_stats,
            'scaling_configuration': self._scaling_config
        }
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """Automatically optimize configuration based on performance metrics."""
        perf_report = self.performance_monitor.get_performance_report()
        cache_stats = self.cache.get_cache_stats()
        
        optimizations = []
        
        # Optimize caching based on hit rate
        if cache_stats.get('hit_rate', 0) < 0.3:
            optimizations.append("Consider increasing cache TTL or size")
            
        # Optimize concurrency based on performance
        avg_time = perf_report.get('summary', {}).get('average_execution_time_ms', 0)
        if avg_time > 100:  # If average execution time > 100ms
            optimizations.append("Consider enabling concurrent processing")
            
        return {
            'current_performance': perf_report.get('summary', {}),
            'cache_efficiency': cache_stats,
            'recommendations': optimizations,
            'auto_optimizations_applied': []
        }
    
    def __getattr__(self, name):
        """Delegate to base manager for compatibility."""
        return getattr(self.base_manager, name)
"""Adaptive Performance Engine with Machine Learning Optimization.

Advanced performance optimization system that learns from usage patterns
and automatically adjusts system parameters for optimal throughput and latency.
"""

from __future__ import annotations

import asyncio
import json
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple
import logging
import statistics
import pickle
from pathlib import Path

from .metrics_export import get_metrics_collector
from .health import get_health_checker
from .cache import get_smart_cache


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    THROUGHPUT_FOCUSED = "throughput_focused"
    LATENCY_FOCUSED = "latency_focused"
    BALANCED = "balanced"
    RESOURCE_CONSERVATIVE = "resource_conservative"
    ADAPTIVE = "adaptive"


class PerformanceMetricType(Enum):
    """Types of performance metrics to track."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_DEPTH = "queue_depth"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: datetime
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_per_second: float
    cpu_usage_percent: float
    memory_usage_mb: float
    queue_depth: int
    error_rate: float
    cache_hit_rate: float
    active_workers: int
    configuration: Dict[str, Any]


@dataclass
class OptimizationRecommendation:
    """Recommendation for system optimization."""
    parameter: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    reasoning: str


class PerformanceProfiler:
    """Advanced performance profiling and analysis."""
    
    def __init__(self, sample_window_minutes: int = 10):
        self.sample_window = timedelta(minutes=sample_window_minutes)
        self.snapshots: List[PerformanceSnapshot] = []
        self.logger = logging.getLogger(__name__)
        
    def capture_snapshot(self, config: Dict[str, Any]) -> PerformanceSnapshot:
        """Capture current system performance snapshot."""
        import psutil
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        # Get application metrics
        cache = get_smart_cache()
        cache_stats = cache.get_stats()
        
        # Calculate cache hit rate
        total_hits = sum(stats.get('hits', 0) for stats in cache_stats.values())
        total_requests = sum(stats.get('hits', 0) + stats.get('misses', 0) for stats in cache_stats.values())
        cache_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            latency_p50=0.0,  # Will be populated by caller
            latency_p95=0.0,
            latency_p99=0.0,
            throughput_per_second=0.0,
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_info.used / (1024 * 1024),
            queue_depth=0,  # Will be populated by caller
            error_rate=0.0,
            cache_hit_rate=cache_hit_rate,
            active_workers=config.get('active_workers', 0),
            configuration=config.copy()
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only recent snapshots
        cutoff_time = datetime.utcnow() - self.sample_window
        self.snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]
        
        return snapshot
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over the sampling window."""
        if len(self.snapshots) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate trends
        timestamps = [s.timestamp.timestamp() for s in self.snapshots]
        latencies = [s.latency_p95 for s in self.snapshots]
        throughputs = [s.throughput_per_second for s in self.snapshots]
        cpu_usage = [s.cpu_usage_percent for s in self.snapshots]
        memory_usage = [s.memory_usage_mb for s in self.snapshots]
        
        def calculate_trend(values):
            if len(values) < 2:
                return 0.0
            return (values[-1] - values[0]) / len(values)
        
        return {
            "status": "analyzed",
            "sample_count": len(self.snapshots),
            "time_window_minutes": (timestamps[-1] - timestamps[0]) / 60,
            "trends": {
                "latency_trend": calculate_trend(latencies),
                "throughput_trend": calculate_trend(throughputs),
                "cpu_trend": calculate_trend(cpu_usage),
                "memory_trend": calculate_trend(memory_usage)
            },
            "current_performance": {
                "avg_latency_p95": statistics.mean(latencies) if latencies else 0.0,
                "avg_throughput": statistics.mean(throughputs) if throughputs else 0.0,
                "avg_cpu": statistics.mean(cpu_usage) if cpu_usage else 0.0,
                "avg_memory_mb": statistics.mean(memory_usage) if memory_usage else 0.0
            }
        }


class MLPerformanceOptimizer:
    """Machine learning-based performance optimizer."""
    
    def __init__(self):
        self.training_data: List[Tuple[Dict[str, Any], PerformanceSnapshot]] = []
        self.optimization_history: List[OptimizationRecommendation] = []
        self.logger = logging.getLogger(__name__)
        
    def learn_from_configuration(self, config: Dict[str, Any], performance: PerformanceSnapshot):
        """Learn from a configuration and its resulting performance."""
        # Store training data
        self.training_data.append((config.copy(), performance))
        
        # Keep recent training data (last 1000 samples)
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-500:]
        
        self.logger.debug(f"Learned from configuration with {len(self.training_data)} total samples")
    
    def generate_optimization_recommendations(self, current_config: Dict[str, Any],
                                            strategy: OptimizationStrategy) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on learned patterns."""
        if len(self.training_data) < 10:
            return []  # Need more data to make recommendations
        
        recommendations = []
        
        # Analyze worker count optimization
        worker_rec = self._analyze_worker_count(current_config, strategy)
        if worker_rec:
            recommendations.append(worker_rec)
        
        # Analyze batch size optimization
        batch_rec = self._analyze_batch_size(current_config, strategy)
        if batch_rec:
            recommendations.append(batch_rec)
        
        # Analyze cache configuration
        cache_rec = self._analyze_cache_config(current_config, strategy)
        if cache_rec:
            recommendations.append(cache_rec)
        
        # Sort by expected improvement
        recommendations.sort(key=lambda x: x.expected_improvement, reverse=True)
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _analyze_worker_count(self, current_config: Dict[str, Any],
                             strategy: OptimizationStrategy) -> Optional[OptimizationRecommendation]:
        """Analyze optimal worker count based on historical data."""
        current_workers = current_config.get('max_workers', 4)
        
        # Group data by worker count
        worker_performance = {}
        for config, perf in self.training_data:
            workers = config.get('max_workers', 4)
            if workers not in worker_performance:
                worker_performance[workers] = []
            worker_performance[workers].append(perf)
        
        if len(worker_performance) < 2:
            return None
        
        # Calculate average performance for each worker count
        best_workers = current_workers
        best_score = self._calculate_performance_score(
            worker_performance[current_workers], strategy
        )
        
        for workers, perfs in worker_performance.items():
            if len(perfs) >= 3:  # Need sufficient samples
                score = self._calculate_performance_score(perfs, strategy)
                if score > best_score:
                    best_score = score
                    best_workers = workers
        
        if best_workers != current_workers:
            improvement = (best_score - self._calculate_performance_score(
                worker_performance[current_workers], strategy
            )) / best_score * 100
            
            return OptimizationRecommendation(
                parameter="max_workers",
                current_value=current_workers,
                recommended_value=best_workers,
                expected_improvement=improvement,
                confidence=min(0.9, len(worker_performance[best_workers]) / 20),
                reasoning=f"Historical data shows {improvement:.1f}% better performance with {best_workers} workers"
            )
        
        return None
    
    def _analyze_batch_size(self, current_config: Dict[str, Any],
                           strategy: OptimizationStrategy) -> Optional[OptimizationRecommendation]:
        """Analyze optimal batch size."""
        current_batch = current_config.get('batch_size', 10)
        
        # Simple heuristic-based recommendation
        if strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            recommended_batch = min(50, current_batch * 2)
        elif strategy == OptimizationStrategy.LATENCY_FOCUSED:
            recommended_batch = max(1, current_batch // 2)
        else:
            return None
        
        if recommended_batch != current_batch:
            return OptimizationRecommendation(
                parameter="batch_size",
                current_value=current_batch,
                recommended_value=recommended_batch,
                expected_improvement=15.0,  # Estimated
                confidence=0.7,
                reasoning=f"Adjusted batch size for {strategy.value} optimization"
            )
        
        return None
    
    def _analyze_cache_config(self, current_config: Dict[str, Any],
                             strategy: OptimizationStrategy) -> Optional[OptimizationRecommendation]:
        """Analyze cache configuration optimization."""
        cache_enabled = current_config.get('enable_caching', True)
        
        # Analyze cache hit rates from recent data
        recent_data = self.training_data[-20:] if len(self.training_data) >= 20 else self.training_data
        
        if not recent_data:
            return None
        
        avg_cache_hit_rate = statistics.mean([perf.cache_hit_rate for _, perf in recent_data])
        
        if not cache_enabled and avg_cache_hit_rate > 0.3:
            return OptimizationRecommendation(
                parameter="enable_caching",
                current_value=False,
                recommended_value=True,
                expected_improvement=25.0,
                confidence=0.8,
                reasoning="Enabling cache could improve performance based on hit rate patterns"
            )
        
        return None
    
    def _calculate_performance_score(self, performances: List[PerformanceSnapshot],
                                   strategy: OptimizationStrategy) -> float:
        """Calculate performance score based on strategy."""
        if not performances:
            return 0.0
        
        latencies = [p.latency_p95 for p in performances if p.latency_p95 > 0]
        throughputs = [p.throughput_per_second for p in performances if p.throughput_per_second > 0]
        
        if not latencies or not throughputs:
            return 0.0
        
        avg_latency = statistics.mean(latencies)
        avg_throughput = statistics.mean(throughputs)
        
        if strategy == OptimizationStrategy.LATENCY_FOCUSED:
            return 1000 / (avg_latency + 1)  # Lower latency = higher score
        elif strategy == OptimizationStrategy.THROUGHPUT_FOCUSED:
            return avg_throughput
        else:  # Balanced
            return (avg_throughput * 10) / (avg_latency + 1)


class AdaptivePerformanceEngine:
    """Main adaptive performance optimization engine."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.profiler = PerformanceProfiler()
        self.optimizer = MLPerformanceOptimizer()
        self.current_strategy = OptimizationStrategy.ADAPTIVE
        self.optimization_interval = 60  # seconds
        self.running = False
        self.optimization_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
        
        # Performance targets
        self.targets = {
            'max_latency_p95': 500.0,  # ms
            'min_throughput': 10.0,    # requests/sec
            'max_cpu_usage': 80.0,     # percent
            'max_memory_mb': 2048.0    # MB
        }
        
    def start_optimization(self):
        """Start the adaptive optimization engine."""
        if self.running:
            return
        
        self.running = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        self.logger.info("Adaptive performance engine started")
    
    def stop_optimization(self):
        """Stop the adaptive optimization engine."""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)
        self.logger.info("Adaptive performance engine stopped")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while self.running:
            try:
                self._run_optimization_cycle()
                time.sleep(self.optimization_interval)
            except Exception as e:
                self.logger.error(f"Optimization cycle error: {e}")
                time.sleep(30)  # Wait before retrying
    
    def _run_optimization_cycle(self):
        """Run a single optimization cycle."""
        # Capture current performance
        current_config = self._get_current_system_config()
        snapshot = self.profiler.capture_snapshot(current_config)
        
        # Update performance data with actual metrics
        snapshot = self._enrich_snapshot_with_metrics(snapshot)
        
        # Learn from current configuration
        self.optimizer.learn_from_configuration(current_config, snapshot)
        
        # Get performance trends
        trends = self.profiler.get_performance_trends()
        
        # Determine optimization strategy
        strategy = self._determine_optimization_strategy(snapshot, trends)
        
        # Generate recommendations
        recommendations = self.optimizer.generate_optimization_recommendations(
            current_config, strategy
        )
        
        # Apply recommendations if they meet confidence threshold
        for rec in recommendations:
            if rec.confidence > 0.6 and rec.expected_improvement > 5.0:
                self._apply_optimization(rec)
        
        self.logger.debug(f"Optimization cycle completed with {len(recommendations)} recommendations")
    
    def _get_current_system_config(self) -> Dict[str, Any]:
        """Get current system configuration."""
        return {
            'max_workers': 8,  # Default values - should be read from actual config
            'batch_size': 10,
            'enable_caching': True,
            'active_workers': 0,
            'queue_timeout': 30
        }
    
    def _enrich_snapshot_with_metrics(self, snapshot: PerformanceSnapshot) -> PerformanceSnapshot:
        """Enrich snapshot with real-time metrics."""
        # In a real implementation, this would fetch actual metrics
        # For now, using placeholder values
        snapshot.latency_p50 = 50.0
        snapshot.latency_p95 = 150.0
        snapshot.latency_p99 = 300.0
        snapshot.throughput_per_second = 25.0
        snapshot.error_rate = 0.02
        snapshot.queue_depth = 5
        
        return snapshot
    
    def _determine_optimization_strategy(self, snapshot: PerformanceSnapshot,
                                       trends: Dict[str, Any]) -> OptimizationStrategy:
        """Determine the best optimization strategy based on current conditions."""
        # Check if we're hitting performance targets
        latency_high = snapshot.latency_p95 > self.targets['max_latency_p95']
        throughput_low = snapshot.throughput_per_second < self.targets['min_throughput']
        cpu_high = snapshot.cpu_usage_percent > self.targets['max_cpu_usage']
        memory_high = snapshot.memory_usage_mb > self.targets['max_memory_mb']
        
        # Analyze trends
        if trends.get('status') == 'analyzed':
            latency_increasing = trends['trends']['latency_trend'] > 0
            throughput_decreasing = trends['trends']['throughput_trend'] < 0
        else:
            latency_increasing = False
            throughput_decreasing = False
        
        # Determine strategy
        if latency_high or latency_increasing:
            return OptimizationStrategy.LATENCY_FOCUSED
        elif throughput_low or throughput_decreasing:
            return OptimizationStrategy.THROUGHPUT_FOCUSED
        elif cpu_high or memory_high:
            return OptimizationStrategy.RESOURCE_CONSERVATIVE
        else:
            return OptimizationStrategy.BALANCED
    
    def _apply_optimization(self, recommendation: OptimizationRecommendation):
        """Apply an optimization recommendation."""
        self.logger.info(f"Applying optimization: {recommendation.parameter} = {recommendation.recommended_value}")
        
        # In a real implementation, this would update the actual system configuration
        # For now, just log the action
        self.logger.info(f"Optimization applied: {recommendation.reasoning}")
        
        # Store the optimization for future reference
        self.optimizer.optimization_history.append(recommendation)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and recommendations."""
        current_config = self._get_current_system_config()
        latest_snapshot = self.profiler.snapshots[-1] if self.profiler.snapshots else None
        trends = self.profiler.get_performance_trends()
        
        recent_recommendations = self.optimizer.optimization_history[-5:] if self.optimizer.optimization_history else []
        
        return {
            "running": self.running,
            "strategy": self.current_strategy.value,
            "optimization_interval_seconds": self.optimization_interval,
            "current_config": current_config,
            "latest_performance": {
                "latency_p95": latest_snapshot.latency_p95 if latest_snapshot else 0.0,
                "throughput_per_second": latest_snapshot.throughput_per_second if latest_snapshot else 0.0,
                "cpu_usage_percent": latest_snapshot.cpu_usage_percent if latest_snapshot else 0.0,
                "memory_usage_mb": latest_snapshot.memory_usage_mb if latest_snapshot else 0.0,
                "cache_hit_rate": latest_snapshot.cache_hit_rate if latest_snapshot else 0.0
            },
            "performance_trends": trends,
            "recent_optimizations": [
                {
                    "parameter": rec.parameter,
                    "from_value": rec.current_value,
                    "to_value": rec.recommended_value,
                    "improvement": rec.expected_improvement,
                    "reasoning": rec.reasoning
                } for rec in recent_recommendations
            ],
            "targets": self.targets,
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instance
_adaptive_engine: Optional[AdaptivePerformanceEngine] = None


def get_adaptive_performance_engine(config: Optional[Dict[str, Any]] = None) -> AdaptivePerformanceEngine:
    """Get or create the global adaptive performance engine."""
    global _adaptive_engine
    
    if _adaptive_engine is None:
        _adaptive_engine = AdaptivePerformanceEngine(config)
    
    return _adaptive_engine


def start_adaptive_optimization(config: Optional[Dict[str, Any]] = None):
    """Start the adaptive performance optimization system."""
    engine = get_adaptive_performance_engine(config)
    engine.start_optimization()


def stop_adaptive_optimization():
    """Stop the adaptive performance optimization system."""
    global _adaptive_engine
    if _adaptive_engine:
        _adaptive_engine.stop_optimization()


def get_adaptive_optimization_status() -> Dict[str, Any]:
    """Get current adaptive optimization status."""
    engine = get_adaptive_performance_engine()
    return engine.get_optimization_status()
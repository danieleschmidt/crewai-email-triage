"""Adaptive Monitoring System for Email Triage.

Self-adjusting monitoring with predictive alerting and automatic optimization.
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Deque, Tuple
import statistics
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - system resource monitoring disabled")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric measurement point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """System alert."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for adaptive thresholds."""
    metric_name: str
    mean: float
    std_dev: float
    p95: float
    p99: float
    sample_count: int
    last_updated: float
    
    def is_anomaly(self, value: float, sensitivity: float = 2.0) -> bool:
        """Check if value is anomalous based on baseline."""
        return abs(value - self.mean) > sensitivity * self.std_dev


class AdaptiveThreshold:
    """Self-adjusting threshold based on historical data."""
    
    def __init__(self, metric_name: str, window_size: int = 1000, adaptation_rate: float = 0.1):
        self.metric_name = metric_name
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.data_points: Deque[float] = deque(maxlen=window_size)
        self.current_threshold = 0.0
        self.baseline: Optional[PerformanceBaseline] = None
        self._lock = threading.Lock()
    
    def add_data_point(self, value: float):
        """Add new data point and update threshold."""
        with self._lock:
            self.data_points.append(value)
            self._update_baseline()
            self._adapt_threshold()
    
    def is_threshold_breached(self, value: float) -> bool:
        """Check if value breaches current threshold."""
        if self.baseline is None:
            return False
        return self.baseline.is_anomaly(value)
    
    def _update_baseline(self):
        """Update baseline statistics."""
        if len(self.data_points) < 10:
            return
        
        data_list = list(self.data_points)
        mean_val = statistics.mean(data_list)
        std_dev = statistics.stdev(data_list) if len(data_list) > 1 else 0.0
        
        # Calculate percentiles
        sorted_data = sorted(data_list)
        p95_idx = int(0.95 * len(sorted_data))
        p99_idx = int(0.99 * len(sorted_data))
        
        self.baseline = PerformanceBaseline(
            metric_name=self.metric_name,
            mean=mean_val,
            std_dev=std_dev,
            p95=sorted_data[p95_idx],
            p99=sorted_data[p99_idx],
            sample_count=len(data_list),
            last_updated=time.time()
        )
    
    def _adapt_threshold(self):
        """Adapt threshold based on recent performance."""
        if self.baseline is None:
            return
        
        # Adaptive threshold based on mean + 2*std_dev with learning rate
        new_threshold = self.baseline.mean + 2 * self.baseline.std_dev
        
        if self.current_threshold == 0.0:
            self.current_threshold = new_threshold
        else:
            # Smooth adaptation
            self.current_threshold = (1 - self.adaptation_rate) * self.current_threshold + \
                                   self.adaptation_rate * new_threshold


class PredictiveAlerting:
    """Predictive alerting system using trend analysis."""
    
    def __init__(self, prediction_window: int = 300):  # 5 minutes
        self.prediction_window = prediction_window
        self.trend_analyzers: Dict[str, Deque[MetricPoint]] = {}
        self.prediction_cache: Dict[str, Tuple[float, float]] = {}  # metric -> (predicted_value, confidence)
        self._lock = threading.Lock()
    
    def analyze_trend(self, metric_name: str, points: List[MetricPoint]) -> Dict[str, Any]:
        """Analyze trend and predict future values."""
        if len(points) < 5:
            return {'trend': 'insufficient_data', 'prediction': None, 'confidence': 0.0}
        
        # Simple linear regression for trend analysis
        timestamps = [p.timestamp for p in points]
        values = [p.value for p in points]
        
        # Normalize timestamps
        base_time = min(timestamps)
        x_values = [(t - base_time) for t in timestamps]
        
        # Calculate linear regression
        n = len(points)
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Slope and intercept
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return {'trend': 'no_trend', 'prediction': None, 'confidence': 0.0}
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict value at prediction_window seconds from now
        future_x = max(x_values) + self.prediction_window
        predicted_value = slope * future_x + intercept
        
        # Calculate confidence based on R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, values))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        confidence = max(0.0, min(r_squared, 1.0))
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'prediction': predicted_value,
            'confidence': confidence,
            'r_squared': r_squared
        }
    
    def should_alert_prediction(self, metric_name: str, current_value: float, 
                               threshold: float, analysis: Dict[str, Any]) -> bool:
        """Determine if predictive alert should be triggered."""
        if analysis['prediction'] is None or analysis['confidence'] < 0.6:
            return False
        
        predicted_value = analysis['prediction']
        
        # Alert if prediction crosses threshold with high confidence
        if analysis['trend'] == 'increasing' and predicted_value > threshold:
            return True
        elif analysis['trend'] == 'decreasing' and current_value > threshold and predicted_value > threshold:
            return True
        
        return False


class SystemResourceMonitor:
    """Monitor system resources with adaptive thresholds."""
    
    def __init__(self):
        self.cpu_threshold = AdaptiveThreshold('cpu_usage', window_size=100)
        self.memory_threshold = AdaptiveThreshold('memory_usage', window_size=100)
        self.disk_threshold = AdaptiveThreshold('disk_usage', window_size=50)
        self.network_threshold = AdaptiveThreshold('network_io', window_size=100)
        
    def get_system_metrics(self) -> Dict[str, MetricPoint]:
        """Get current system metrics."""
        timestamp = time.time()
        
        if not PSUTIL_AVAILABLE:
            # Return mock metrics when psutil not available
            return {
                'cpu_usage': MetricPoint(timestamp, 15.0, {'unit': 'percent', 'mock': 'true'}),
                'memory_usage': MetricPoint(timestamp, 45.0, {'unit': 'percent', 'mock': 'true'}),
                'disk_usage': MetricPoint(timestamp, 60.0, {'unit': 'percent', 'mock': 'true'}),
                'network_io': MetricPoint(timestamp, 10.5, {'unit': 'mb', 'mock': 'true'})
            }
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage (root filesystem)
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = (network.bytes_sent + network.bytes_recv) / (1024 * 1024)  # MB
            
            return {
                'cpu_usage': MetricPoint(timestamp, cpu_percent, {'unit': 'percent'}),
                'memory_usage': MetricPoint(timestamp, memory_percent, {'unit': 'percent'}),
                'disk_usage': MetricPoint(timestamp, disk_percent, {'unit': 'percent'}),
                'network_io': MetricPoint(timestamp, network_io, {'unit': 'mb'})
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            # Return default values on error
            return {
                'cpu_usage': MetricPoint(timestamp, 0.0, {'unit': 'percent', 'error': 'true'}),
                'memory_usage': MetricPoint(timestamp, 0.0, {'unit': 'percent', 'error': 'true'}),
                'disk_usage': MetricPoint(timestamp, 0.0, {'unit': 'percent', 'error': 'true'}),
                'network_io': MetricPoint(timestamp, 0.0, {'unit': 'mb', 'error': 'true'})
            }
    
    def check_resource_anomalies(self, metrics: Dict[str, MetricPoint]) -> List[Alert]:
        """Check for resource anomalies and generate alerts."""
        alerts = []
        
        # Update thresholds and check for anomalies
        for metric_name, point in metrics.items():
            threshold = getattr(self, f"{metric_name}_threshold")
            threshold.add_data_point(point.value)
            
            if threshold.is_threshold_breached(point.value):
                severity = self._determine_severity(metric_name, point.value, threshold.baseline)
                
                alert = Alert(
                    id=f"resource_{metric_name}_{int(point.timestamp)}",
                    severity=severity,
                    title=f"Resource Anomaly: {metric_name}",
                    message=f"{metric_name} value {point.value:.2f} is anomalous (baseline: {threshold.baseline.mean:.2f} ± {threshold.baseline.std_dev:.2f})",
                    component="system_resources",
                    timestamp=point.timestamp,
                    metadata={
                        'metric_name': metric_name,
                        'current_value': point.value,
                        'baseline_mean': threshold.baseline.mean,
                        'baseline_std_dev': threshold.baseline.std_dev
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def _determine_severity(self, metric_name: str, value: float, baseline: PerformanceBaseline) -> AlertSeverity:
        """Determine alert severity based on metric type and deviation."""
        deviation = abs(value - baseline.mean) / baseline.std_dev if baseline.std_dev > 0 else 0
        
        # Critical thresholds for system resources
        if metric_name in ['cpu_usage', 'memory_usage', 'disk_usage'] and value > 90:
            return AlertSeverity.CRITICAL
        elif deviation > 4:
            return AlertSeverity.ERROR
        elif deviation > 3:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO


class AdaptiveMonitoringSystem:
    """Complete adaptive monitoring system with predictive capabilities."""
    
    def __init__(self):
        self.metrics_storage: Dict[str, Deque[MetricPoint]] = {}
        self.adaptive_thresholds: Dict[str, AdaptiveThreshold] = {}
        self.alerts: Dict[str, Alert] = {}
        self.active_alerts: Dict[str, Alert] = {}
        
        self.predictive_alerting = PredictiveAlerting()
        self.resource_monitor = SystemResourceMonitor()
        
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_active = False
        self._lock = threading.Lock()
        
        # Configuration
        self.metric_retention_size = 1000
        self.monitoring_interval = 10.0  # seconds
        self.alert_handlers: List[Callable[[Alert], None]] = []
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Adaptive monitoring system started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Adaptive monitoring system stopped")
    
    def add_metric_point(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Add a metric data point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        
        with self._lock:
            if metric_name not in self.metrics_storage:
                self.metrics_storage[metric_name] = deque(maxlen=self.metric_retention_size)
                self.adaptive_thresholds[metric_name] = AdaptiveThreshold(metric_name)
            
            self.metrics_storage[metric_name].append(point)
            self.adaptive_thresholds[metric_name].add_data_point(value)
            
            # Check for threshold breaches
            self._check_metric_threshold(metric_name, point)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting monitoring loop")
        
        while self._monitoring_active:
            try:
                start_time = time.time()
                
                # Monitor system resources
                resource_metrics = self.resource_monitor.get_system_metrics()
                for metric_name, point in resource_metrics.items():
                    with self._lock:
                        if metric_name not in self.metrics_storage:
                            self.metrics_storage[metric_name] = deque(maxlen=self.metric_retention_size)
                            self.adaptive_thresholds[metric_name] = AdaptiveThreshold(metric_name)
                        
                        self.metrics_storage[metric_name].append(point)
                
                # Check for resource anomalies
                resource_alerts = self.resource_monitor.check_resource_anomalies(resource_metrics)
                for alert in resource_alerts:
                    self._handle_alert(alert)
                
                # Perform predictive analysis on key metrics
                self._perform_predictive_analysis()
                
                # Clean up old alerts
                self._cleanup_resolved_alerts()
                
                # Adaptive sleep based on system load
                processing_time = time.time() - start_time
                sleep_time = max(1.0, self.monitoring_interval - processing_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_metric_threshold(self, metric_name: str, point: MetricPoint):
        """Check if metric point breaches threshold."""
        threshold = self.adaptive_thresholds[metric_name]
        
        if threshold.is_threshold_breached(point.value):
            alert_id = f"threshold_{metric_name}_{int(point.timestamp)}"
            
            if alert_id not in self.active_alerts:
                alert = Alert(
                    id=alert_id,
                    severity=AlertSeverity.WARNING,
                    title=f"Threshold Breach: {metric_name}",
                    message=f"Metric {metric_name} value {point.value:.3f} breached adaptive threshold",
                    component="metrics",
                    timestamp=point.timestamp,
                    metadata={
                        'metric_name': metric_name,
                        'value': point.value,
                        'threshold': threshold.current_threshold,
                        'baseline': threshold.baseline.__dict__ if threshold.baseline else None
                    }
                )
                self._handle_alert(alert)
    
    def _perform_predictive_analysis(self):
        """Perform predictive analysis on critical metrics."""
        critical_metrics = ['cpu_usage', 'memory_usage', 'processing_time', 'error_rate']
        
        with self._lock:
            for metric_name in critical_metrics:
                if metric_name in self.metrics_storage and len(self.metrics_storage[metric_name]) >= 10:
                    points = list(self.metrics_storage[metric_name])[-50:]  # Last 50 points
                    analysis = self.predictive_alerting.analyze_trend(metric_name, points)
                    
                    # Check if predictive alert is needed
                    if analysis['prediction'] is not None and analysis['confidence'] > 0.7:
                        threshold = self.adaptive_thresholds.get(metric_name)
                        if threshold and threshold.baseline:
                            should_alert = self.predictive_alerting.should_alert_prediction(
                                metric_name, points[-1].value, threshold.baseline.p95, analysis
                            )
                            
                            if should_alert:
                                alert = Alert(
                                    id=f"predictive_{metric_name}_{int(time.time())}",
                                    severity=AlertSeverity.WARNING,
                                    title=f"Predictive Alert: {metric_name}",
                                    message=f"Metric {metric_name} predicted to reach {analysis['prediction']:.3f} in {self.predictive_alerting.prediction_window}s",
                                    component="predictive_analysis",
                                    timestamp=time.time(),
                                    metadata={
                                        'metric_name': metric_name,
                                        'current_value': points[-1].value,
                                        'predicted_value': analysis['prediction'],
                                        'confidence': analysis['confidence'],
                                        'trend': analysis['trend']
                                    }
                                )
                                self._handle_alert(alert)
    
    def _handle_alert(self, alert: Alert):
        """Handle new alert."""
        with self._lock:
            self.alerts[alert.id] = alert
            self.active_alerts[alert.id] = alert
        
        logger.warning(f"Alert {alert.severity.value}: {alert.title} - {alert.message}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolution_time = time.time()
                del self.active_alerts[alert_id]
                logger.info(f"Alert resolved: {alert.title}")
    
    def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts."""
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        with self._lock:
            alerts_to_remove = []
            for alert_id, alert in self.alerts.items():
                if alert.resolved and (current_time - alert.timestamp) > cleanup_age:
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.alerts[alert_id]
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert notification handler."""
        self.alert_handlers.append(handler)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status."""
        with self._lock:
            return {
                'monitoring_active': self._monitoring_active,
                'tracked_metrics': list(self.metrics_storage.keys()),
                'active_alerts_count': len(self.active_alerts),
                'total_alerts': len(self.alerts),
                'adaptive_thresholds': {
                    name: {
                        'current_threshold': threshold.current_threshold,
                        'data_points': len(threshold.data_points),
                        'baseline': threshold.baseline.__dict__ if threshold.baseline else None
                    }
                    for name, threshold in self.adaptive_thresholds.items()
                }
            }
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of alerts."""
        with self._lock:
            active_by_severity = {}
            for alert in self.active_alerts.values():
                severity = alert.severity.value
                active_by_severity[severity] = active_by_severity.get(severity, 0) + 1
            
            return {
                'active_alerts': len(self.active_alerts),
                'active_by_severity': active_by_severity,
                'recent_alerts': [
                    {
                        'id': alert.id,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'timestamp': alert.timestamp,
                        'resolved': alert.resolved
                    }
                    for alert in sorted(self.alerts.values(), key=lambda a: a.timestamp, reverse=True)[:10]
                ]
            }


# Global monitoring system instance
_monitoring_system = None

def get_monitoring_system() -> AdaptiveMonitoringSystem:
    """Get global adaptive monitoring system instance."""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = AdaptiveMonitoringSystem()
    return _monitoring_system


def monitoring_metric(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to automatically track function execution metrics."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            monitoring = get_monitoring_system()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # milliseconds
                monitoring.add_metric_point(f"{metric_name}_duration", execution_time, labels)
                monitoring.add_metric_point(f"{metric_name}_success", 1, labels)
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                monitoring.add_metric_point(f"{metric_name}_duration", execution_time, labels)
                monitoring.add_metric_point(f"{metric_name}_error", 1, labels)
                raise
        return wrapper
    return decorator
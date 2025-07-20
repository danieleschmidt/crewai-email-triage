"""Metrics export functionality for Prometheus and OpenTelemetry integration.

This module provides comprehensive metrics collection and export capabilities
for monitoring the email triage pipeline in production environments.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler


@dataclass
class MetricsConfig:
    """Configuration for metrics collection and export."""
    
    enabled: bool = True
    export_port: int = 8080
    export_path: str = "/metrics"
    namespace: str = "crewai_email_triage"
    
    @classmethod
    def from_environment(cls) -> MetricsConfig:
        """Create configuration from environment variables."""
        return cls(
            enabled=os.environ.get("METRICS_ENABLED", "true").lower() == "true",
            export_port=int(os.environ.get("METRICS_EXPORT_PORT", "8080")),
            export_path=os.environ.get("METRICS_EXPORT_PATH", "/metrics"),
            namespace=os.environ.get("METRICS_NAMESPACE", "crewai_email_triage"),
        )
    
    def __post_init__(self):
        """Validate configuration values."""
        if not (1 <= self.export_port <= 65535):
            raise ValueError(f"Invalid export_port: {self.export_port}. Must be between 1 and 65535.")
        
        if not self.export_path.startswith("/"):
            raise ValueError(f"Invalid export_path: {self.export_path}. Must start with '/'.")


class MetricsCollector:
    """Thread-safe metrics collector for counters, gauges, and histograms."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a counter metric."""
        with self._lock:
            self._counters[name] += value
    
    def get_counter(self, name: str) -> int:
        """Get the current value of a counter."""
        with self._lock:
            return self._counters[name]
    
    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge metric to a specific value."""
        with self._lock:
            self._gauges[name] = value
    
    def increment_gauge(self, name: str, value: float = 1) -> None:
        """Increment a gauge metric by the specified value."""
        with self._lock:
            self._gauges[name] += value
    
    def get_gauge(self, name: str) -> float:
        """Get the current value of a gauge."""
        with self._lock:
            return self._gauges[name]
    
    def record_histogram(self, name: str, value: float) -> None:
        """Record a value in a histogram metric."""
        with self._lock:
            self._histograms[name].append(value)
    
    def get_histogram(self, name: str) -> List[float]:
        """Get all recorded values for a histogram."""
        with self._lock:
            return self._histograms[name].copy()
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics organized by type."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: v.copy() for k, v in self._histograms.items()},
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics to their initial state."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


class PrometheusExporter:
    """Export metrics in Prometheus format."""
    
    def __init__(self, collector: MetricsCollector, namespace: str = "crewai_email_triage"):
        """Initialize the Prometheus exporter."""
        self.collector = collector
        self.namespace = namespace
    
    def export(self) -> str:
        """Export all metrics in Prometheus format."""
        output = []
        all_metrics = self.collector.get_all_metrics()
        
        # Export counters
        for name, value in all_metrics["counters"].items():
            metric_name = f"{self.namespace}_{name}"
            output.append(f"# TYPE {metric_name} counter")
            output.append(f"{metric_name} {value}")
        
        # Export gauges
        for name, value in all_metrics["gauges"].items():
            metric_name = f"{self.namespace}_{name}"
            output.append(f"# TYPE {metric_name} gauge")
            output.append(f"{metric_name} {value}")
        
        # Export histograms (simplified - just count and sum)
        for name, values in all_metrics["histograms"].items():
            if values:
                metric_name = f"{self.namespace}_{name}"
                count = len(values)
                total = sum(values)
                output.append(f"# TYPE {metric_name} histogram")
                output.append(f"{metric_name}_count {count}")
                output.append(f"{metric_name}_sum {total}")
        
        return "\n".join(output) + "\n"


class MetricsEndpoint:
    """HTTP endpoint for serving metrics."""
    
    def __init__(self, exporter: PrometheusExporter, config: MetricsConfig):
        """Initialize the metrics endpoint."""
        self.exporter = exporter
        self.config = config
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
    
    def start(self) -> None:
        """Start the metrics HTTP server."""
        if not self.config.enabled:
            return
        
        handler = self._create_handler()
        self._server = HTTPServer(("", self.config.export_port), handler)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
    
    def stop(self) -> None:
        """Stop the metrics HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        
        if self._server_thread:
            self._server_thread.join(timeout=5)
            self._server_thread = None
    
    def _create_handler(self):
        """Create HTTP request handler for metrics endpoint."""
        exporter = self.exporter
        config = self.config
        
        class MetricsHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == config.export_path:
                    metrics_output = exporter.export()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(metrics_output.encode("utf-8"))
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Suppress default HTTP server logging
                pass
        
        return MetricsHandler


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def export_metrics_to_prometheus_format() -> str:
    """Export current metrics in Prometheus format."""
    collector = get_metrics_collector()
    exporter = PrometheusExporter(collector)
    return exporter.export()
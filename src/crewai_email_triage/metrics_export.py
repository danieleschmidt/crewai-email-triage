"""Metrics export functionality for Prometheus and OpenTelemetry integration.

This module provides comprehensive metrics collection and export capabilities
for monitoring the email triage pipeline in production environments.
"""

from __future__ import annotations

import threading
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
    histogram_max_size: int = 1000  # Maximum number of values to keep in histograms
    
    @classmethod
    def from_environment(cls) -> MetricsConfig:
        """Create configuration from environment variables."""
        from .env_config import get_metrics_config
        
        env_config = get_metrics_config()
        return cls(
            enabled=env_config.enabled,
            export_port=env_config.export_port,
            export_path=env_config.export_path,
            namespace=env_config.namespace,
            histogram_max_size=env_config.histogram_max_size,
        )
    
    def __post_init__(self):
        """Validate configuration values."""
        if not (1 <= self.export_port <= 65535):
            raise ValueError(f"Invalid export_port: {self.export_port}. Must be between 1 and 65535.")
        
        if not self.export_path.startswith("/"):
            raise ValueError(f"Invalid export_path: {self.export_path}. Must start with '/'.")


class MetricsCollector:
    """Thread-safe metrics collector for counters, gauges, and histograms."""
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """Initialize the metrics collector."""
        self._config = config or MetricsConfig()
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self._config.histogram_max_size))
    
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
            return list(self._histograms[name])
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics organized by type."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {k: list(v) for k, v in self._histograms.items()},
            }
    
    def reset_metrics(self) -> None:
        """Reset all metrics to their initial state."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistical summary of a histogram for Prometheus export."""
        with self._lock:
            values = list(self._histograms[name])
            if not values:
                return {"count": 0, "sum": 0.0}
            
            return {
                "count": len(values),
                "sum": sum(values),
            }


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
        
        # Export histograms (using efficient stats calculation)
        for name in all_metrics["histograms"].keys():
            stats = self.collector.get_histogram_stats(name)
            if stats["count"] > 0:
                metric_name = f"{self.namespace}_{name}"
                output.append(f"# TYPE {metric_name} histogram")
                output.append(f"{metric_name}_count {int(stats['count'])}")
                output.append(f"{metric_name}_sum {stats['sum']}")
        
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
                """Handle GET requests with proper validation and security headers."""
                try:
                    # Validate path
                    if self.path == config.export_path:
                        metrics_output = exporter.export()
                        self.send_response(200)
                        self._send_security_headers()
                        self.send_header("Content-Type", "text/plain; charset=utf-8")
                        self.send_header("Content-Length", str(len(metrics_output.encode("utf-8"))))
                        self.end_headers()
                        self.wfile.write(metrics_output.encode("utf-8"))
                    elif self.path == "/health":
                        # Health check endpoint for liveness probes
                        health_status = self._get_health_status()
                        health_response = health_status["response"]
                        status_code = health_status["status_code"]
                        
                        self.send_response(status_code)
                        self._send_security_headers()
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Length", str(len(health_response.encode("utf-8"))))
                        self.end_headers()
                        self.wfile.write(health_response.encode("utf-8"))
                    elif self.path == "/ready":
                        # Readiness check endpoint for readiness probes
                        readiness_status = self._get_readiness_status()
                        readiness_response = readiness_status["response"]
                        status_code = readiness_status["status_code"]
                        
                        self.send_response(status_code)
                        self._send_security_headers()
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Length", str(len(readiness_response.encode("utf-8"))))
                        self.end_headers()
                        self.wfile.write(readiness_response.encode("utf-8"))
                    else:
                        self._send_error_response(404, "Not Found")
                except Exception:
                    self._send_error_response(500, "Internal Server Error")
            
            def do_POST(self):
                """Reject POST requests with proper error response."""
                self._send_error_response(405, "Method Not Allowed", {"Allow": "GET"})
            
            def do_PUT(self):
                """Reject PUT requests with proper error response."""
                self._send_error_response(405, "Method Not Allowed", {"Allow": "GET"})
            
            def do_DELETE(self):
                """Reject DELETE requests with proper error response."""
                self._send_error_response(405, "Method Not Allowed", {"Allow": "GET"})
            
            def do_HEAD(self):
                """Handle HEAD requests (same as GET but without body)."""
                try:
                    if self.path == config.export_path or self.path == "/health" or self.path == "/ready":
                        # For HEAD requests, only return status based on health/readiness
                        status_code = 200
                        if self.path == "/health":
                            health_status = self._get_health_status()
                            status_code = health_status["status_code"]
                        elif self.path == "/ready":
                            readiness_status = self._get_readiness_status()
                            status_code = readiness_status["status_code"]
                        
                        self.send_response(status_code)
                        self._send_security_headers()
                        if self.path == config.export_path:
                            self.send_header("Content-Type", "text/plain; charset=utf-8")
                        else:
                            self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.end_headers()
                    else:
                        self._send_error_response(404, "Not Found")
                except Exception:
                    self._send_error_response(500, "Internal Server Error")
            
            def _send_security_headers(self):
                """Send standard security headers."""
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("X-XSS-Protection", "1; mode=block")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
            
            def _send_error_response(self, code: int, message: str, extra_headers: dict = None):
                """Send standardized error responses with security headers."""
                try:
                    self.send_response(code)
                    self._send_security_headers()
                    if extra_headers:
                        for header, value in extra_headers.items():
                            self.send_header(header, value)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.end_headers()
                    if hasattr(self, 'wfile'):
                        self.wfile.write(f"{code} {message}\n".encode("utf-8"))
                except Exception:
                    pass  # Avoid recursive errors
            
            def log_message(self, format, *args):
                # Suppress default HTTP server logging to avoid log pollution
                pass
            
            def _get_health_status(self) -> dict:
                """Get health status for liveness probe (basic service availability)."""
                try:
                    # Basic health check - service is running and can serve requests
                    import time
                    current_time = time.time()
                    
                    response = {
                        "status": "healthy",
                        "service": "email-triage-metrics",
                        "timestamp": int(current_time),
                        "version": "1.0"
                    }
                    
                    return {
                        "response": f"{response}\n".replace("'", '"'),
                        "status_code": 200
                    }
                except Exception:
                    error_response = {
                        "status": "unhealthy",
                        "service": "email-triage-metrics",
                        "error": "Internal service error"
                    }
                    return {
                        "response": f"{error_response}\n".replace("'", '"'),
                        "status_code": 503
                    }
            
            def _get_readiness_status(self) -> dict:
                """Get readiness status for readiness probe (service ready to handle requests)."""
                try:
                    # Check if metrics collector is accessible and functional
                    test_metrics = exporter.collector.get_all_metrics()
                    
                    # Validate that metrics are available
                    if test_metrics is None:
                        raise RuntimeError("Metrics collector returned None")
                    
                    # Check if we can generate metrics export
                    _ = exporter.export()
                    
                    import time
                    current_time = time.time()
                    
                    response = {
                        "status": "ready",
                        "service": "email-triage-metrics",
                        "timestamp": int(current_time),
                        "checks": {
                            "metrics_collector": "ok",
                            "prometheus_export": "ok"
                        }
                    }
                    
                    return {
                        "response": f"{response}\n".replace("'", '"'),
                        "status_code": 200
                    }
                except Exception:
                    import time
                    current_time = time.time()
                    
                    error_response = {
                        "status": "not_ready",
                        "service": "email-triage-metrics",
                        "timestamp": int(current_time),
                        "error": "Service dependencies not ready",
                        "checks": {
                            "metrics_collector": "error",
                            "prometheus_export": "error"
                        }
                    }
                    return {
                        "response": f"{error_response}\n".replace("'", '"'),
                        "status_code": 503
                    }
            
            def version_string(self):
                # Don't reveal server version for security
                return "EmailTriageMetrics/1.0"
        
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
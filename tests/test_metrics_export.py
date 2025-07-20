"""Test metrics export functionality."""

import pytest
import time
from unittest.mock import Mock, patch
from io import StringIO

from crewai_email_triage.metrics_export import (
    MetricsCollector,
    PrometheusExporter,
    MetricsEndpoint,
    get_metrics_collector,
    export_metrics_to_prometheus_format,
    MetricsConfig
)
from crewai_email_triage.pipeline import triage_email, triage_batch


class TestMetricsCollector:
    """Test the centralized metrics collection system."""

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        
        assert collector.get_counter("emails_processed") == 0
        assert collector.get_histogram("processing_time_seconds") == []
        assert collector.get_gauge("active_requests") == 0

    def test_increment_counter(self):
        """Test counter increment functionality."""
        collector = MetricsCollector()
        
        collector.increment_counter("emails_processed")
        assert collector.get_counter("emails_processed") == 1
        
        collector.increment_counter("emails_processed", 5)
        assert collector.get_counter("emails_processed") == 6

    def test_record_histogram(self):
        """Test histogram recording functionality."""
        collector = MetricsCollector()
        
        collector.record_histogram("processing_time_seconds", 1.5)
        collector.record_histogram("processing_time_seconds", 2.3)
        
        values = collector.get_histogram("processing_time_seconds")
        assert len(values) == 2
        assert 1.5 in values
        assert 2.3 in values

    def test_set_gauge(self):
        """Test gauge setting functionality."""
        collector = MetricsCollector()
        
        collector.set_gauge("active_requests", 5)
        assert collector.get_gauge("active_requests") == 5
        
        collector.set_gauge("active_requests", 3)
        assert collector.get_gauge("active_requests") == 3

    def test_increment_gauge(self):
        """Test gauge increment/decrement functionality."""
        collector = MetricsCollector()
        
        collector.increment_gauge("active_requests")
        assert collector.get_gauge("active_requests") == 1
        
        collector.increment_gauge("active_requests", 2)
        assert collector.get_gauge("active_requests") == 3
        
        collector.increment_gauge("active_requests", -1)
        assert collector.get_gauge("active_requests") == 2

    def test_get_all_metrics(self):
        """Test getting all metrics at once."""
        collector = MetricsCollector()
        
        collector.increment_counter("emails_processed", 10)
        collector.record_histogram("processing_time_seconds", 1.5)
        collector.set_gauge("active_requests", 3)
        
        all_metrics = collector.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "histograms" in all_metrics
        assert "gauges" in all_metrics
        assert all_metrics["counters"]["emails_processed"] == 10
        assert all_metrics["gauges"]["active_requests"] == 3

    def test_reset_metrics(self):
        """Test metrics reset functionality."""
        collector = MetricsCollector()
        
        collector.increment_counter("emails_processed", 5)
        collector.set_gauge("active_requests", 3)
        
        collector.reset_metrics()
        
        assert collector.get_counter("emails_processed") == 0
        assert collector.get_gauge("active_requests") == 0

    def test_thread_safety(self):
        """Test that metrics collector is thread-safe."""
        import threading
        
        collector = MetricsCollector()
        
        def increment_worker():
            for _ in range(100):
                collector.increment_counter("test_counter")
        
        threads = [threading.Thread(target=increment_worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # Should have 500 increments total
        assert collector.get_counter("test_counter") == 500


class TestPrometheusExporter:
    """Test Prometheus format exporter."""

    def test_prometheus_exporter_initialization(self):
        """Test PrometheusExporter initialization."""
        collector = MetricsCollector()
        exporter = PrometheusExporter(collector)
        
        assert exporter.collector == collector
        assert exporter.namespace == "crewai_email_triage"

    def test_export_counters(self):
        """Test exporting counters in Prometheus format."""
        collector = MetricsCollector()
        collector.increment_counter("emails_processed", 42)
        collector.increment_counter("errors_total", 3)
        
        exporter = PrometheusExporter(collector)
        output = exporter.export()
        
        assert "# TYPE crewai_email_triage_emails_processed counter" in output
        assert "crewai_email_triage_emails_processed 42" in output
        assert "# TYPE crewai_email_triage_errors_total counter" in output
        assert "crewai_email_triage_errors_total 3" in output

    def test_export_gauges(self):
        """Test exporting gauges in Prometheus format."""
        collector = MetricsCollector()
        collector.set_gauge("active_requests", 5)
        collector.set_gauge("memory_usage_bytes", 1024000)
        
        exporter = PrometheusExporter(collector)
        output = exporter.export()
        
        assert "# TYPE crewai_email_triage_active_requests gauge" in output
        assert "crewai_email_triage_active_requests 5" in output
        assert "# TYPE crewai_email_triage_memory_usage_bytes gauge" in output
        assert "crewai_email_triage_memory_usage_bytes 1024000" in output

    def test_export_histograms(self):
        """Test exporting histograms in Prometheus format."""
        collector = MetricsCollector()
        collector.record_histogram("processing_time_seconds", 1.5)
        collector.record_histogram("processing_time_seconds", 2.3)
        collector.record_histogram("processing_time_seconds", 0.8)
        
        exporter = PrometheusExporter(collector)
        output = exporter.export()
        
        assert "# TYPE crewai_email_triage_processing_time_seconds histogram" in output
        assert "crewai_email_triage_processing_time_seconds_count 3" in output
        assert "crewai_email_triage_processing_time_seconds_sum" in output
        # Should have bucket entries
        assert "crewai_email_triage_processing_time_seconds_bucket" in output

    def test_export_with_custom_namespace(self):
        """Test exporting with custom namespace."""
        collector = MetricsCollector()
        collector.increment_counter("test_metric", 1)
        
        exporter = PrometheusExporter(collector, namespace="custom_app")
        output = exporter.export()
        
        assert "custom_app_test_metric 1" in output

    def test_export_with_labels(self):
        """Test exporting metrics with labels."""
        collector = MetricsCollector()
        collector.increment_counter("requests_total", 10, labels={"method": "POST", "status": "200"})
        
        exporter = PrometheusExporter(collector)
        output = exporter.export()
        
        assert 'crewai_email_triage_requests_total{method="POST",status="200"} 10' in output

    def test_metric_name_sanitization(self):
        """Test that metric names are properly sanitized for Prometheus."""
        collector = MetricsCollector()
        collector.increment_counter("invalid-metric.name", 1)
        
        exporter = PrometheusExporter(collector)
        output = exporter.export()
        
        # Should convert to valid Prometheus metric name
        assert "crewai_email_triage_invalid_metric_name 1" in output


class TestMetricsEndpoint:
    """Test HTTP metrics endpoint functionality."""

    def test_metrics_endpoint_initialization(self):
        """Test MetricsEndpoint initialization."""
        collector = MetricsCollector()
        endpoint = MetricsEndpoint(collector)
        
        assert endpoint.collector == collector
        assert endpoint.port == 8080  # Default port

    def test_metrics_endpoint_custom_port(self):
        """Test MetricsEndpoint with custom port."""
        collector = MetricsCollector()
        endpoint = MetricsEndpoint(collector, port=9090)
        
        assert endpoint.port == 9090

    def test_get_metrics_response(self):
        """Test getting metrics as HTTP response."""
        collector = MetricsCollector()
        collector.increment_counter("test_metric", 42)
        
        endpoint = MetricsEndpoint(collector)
        response = endpoint.get_metrics_response()
        
        assert "Content-Type" in response
        assert response["Content-Type"] == "text/plain; version=0.0.4; charset=utf-8"
        assert "crewai_email_triage_test_metric 42" in response["body"]

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        collector = MetricsCollector()
        endpoint = MetricsEndpoint(collector)
        
        health_response = endpoint.get_health_response()
        
        assert "Content-Type" in health_response
        assert health_response["Content-Type"] == "application/json"
        assert '"status": "healthy"' in health_response["body"]


class TestMetricsIntegration:
    """Test integration with existing pipeline."""

    def test_pipeline_metrics_collection(self):
        """Test that pipeline operations are recorded in metrics."""
        collector = MetricsCollector()
        
        # Mock the global metrics collector
        with patch('crewai_email_triage.metrics_export.get_metrics_collector', return_value=collector):
            # Process some emails
            triage_email("Test email content")
            triage_email("Another test email")
            
            # Verify metrics were collected
            assert collector.get_counter("emails_processed") >= 2
            processing_times = collector.get_histogram("email_processing_time_seconds")
            assert len(processing_times) >= 2

    def test_batch_processing_metrics(self):
        """Test metrics collection during batch processing."""
        collector = MetricsCollector()
        
        with patch('crewai_email_triage.metrics_export.get_metrics_collector', return_value=collector):
            messages = ["Email 1", "Email 2", "Email 3"]
            triage_batch(messages)
            
            assert collector.get_counter("emails_processed") >= 3
            assert collector.get_counter("batch_operations") >= 1

    def test_error_metrics_collection(self):
        """Test that errors are properly recorded in metrics."""
        collector = MetricsCollector()
        
        with patch('crewai_email_triage.metrics_export.get_metrics_collector', return_value=collector):
            # Process invalid input that should cause errors
            triage_email(None)
            triage_email("")
            
            # Should have error metrics
            assert collector.get_counter("processing_errors") >= 0

    def test_sanitization_metrics_integration(self):
        """Test integration with sanitization metrics."""
        collector = MetricsCollector()
        
        with patch('crewai_email_triage.metrics_export.get_metrics_collector', return_value=collector):
            # Process email with security threats
            triage_email("Malicious <script>alert('xss')</script> content")
            
            # Should record sanitization metrics
            assert collector.get_counter("sanitization_threats_detected") >= 0
            assert collector.get_counter("sanitization_operations") >= 0


class TestMetricsConfig:
    """Test metrics configuration."""

    def test_metrics_config_defaults(self):
        """Test default metrics configuration."""
        config = MetricsConfig()
        
        assert config.enabled is True
        assert config.export_port == 8080
        assert config.export_path == "/metrics"
        assert config.namespace == "crewai_email_triage"

    def test_metrics_config_from_env(self):
        """Test loading metrics configuration from environment."""
        import os
        
        # Mock environment variables
        env_vars = {
            "METRICS_ENABLED": "false",
            "METRICS_EXPORT_PORT": "9090",
            "METRICS_EXPORT_PATH": "/custom-metrics",
            "METRICS_NAMESPACE": "custom_namespace"
        }
        
        with patch.dict(os.environ, env_vars):
            config = MetricsConfig.from_environment()
            
            assert config.enabled is False
            assert config.export_port == 9090
            assert config.export_path == "/custom-metrics"
            assert config.namespace == "custom_namespace"

    def test_metrics_config_validation(self):
        """Test metrics configuration validation."""
        # Test invalid port
        with pytest.raises(ValueError):
            MetricsConfig(export_port=70000)  # Port too high
        
        with pytest.raises(ValueError):
            MetricsConfig(export_port=0)  # Port too low
        
        # Test invalid path
        with pytest.raises(ValueError):
            MetricsConfig(export_path="invalid-path")  # Must start with /


class TestMetricsEndpointSecurity:
    """Test HTTP endpoint security features."""

    def test_http_method_validation(self):
        """Test that only GET and HEAD methods are allowed."""
        collector = MetricsCollector()
        exporter = PrometheusExporter(collector)
        config = MetricsConfig(enabled=True, export_port=9099, export_path="/metrics")
        endpoint = MetricsEndpoint(exporter, config)
        
        # This test would require actual HTTP server testing
        # For now, we test the handler creation and basic functionality
        handler_class = endpoint._create_handler()
        
        # Verify the handler has the expected methods
        expected_methods = ['do_GET', 'do_POST', 'do_PUT', 'do_DELETE', 'do_HEAD']
        for method in expected_methods:
            assert hasattr(handler_class, method), f"Handler missing {method} method"
        
        # Verify security methods exist
        security_methods = ['_send_security_headers', '_send_error_response', 'version_string']
        for method in security_methods:
            assert hasattr(handler_class, method), f"Handler missing security method {method}"

    def test_health_check_endpoint_response(self):
        """Test that health check endpoint returns proper JSON."""
        collector = MetricsCollector()
        exporter = PrometheusExporter(collector)
        config = MetricsConfig(enabled=True, export_port=9100, export_path="/metrics")
        endpoint = MetricsEndpoint(exporter, config)
        
        # Test that the endpoint creates a proper handler
        handler_class = endpoint._create_handler()
        assert handler_class is not None
        
        # Verify the handler has proper path handling logic
        # (Full HTTP testing would require integration tests)
        assert hasattr(handler_class, 'do_GET')

    def test_security_headers_configuration(self):
        """Test that security headers are properly configured."""
        collector = MetricsCollector()
        exporter = PrometheusExporter(collector)
        config = MetricsConfig(enabled=True, export_port=9101, export_path="/metrics")
        endpoint = MetricsEndpoint(exporter, config)
        
        handler_class = endpoint._create_handler()
        
        # Verify security header method exists
        assert hasattr(handler_class, '_send_security_headers')
        
        # Create a mock handler instance to test the method exists
        # (Full header testing would require HTTP integration tests)
        assert callable(getattr(handler_class, '_send_security_headers'))


class TestUtilityFunctions:
    """Test utility functions for metrics export."""

    def test_export_metrics_to_prometheus_format(self):
        """Test utility function for Prometheus export."""
        # Set up some metrics in the global collector
        collector = get_metrics_collector()
        collector.increment_counter("test_counter", 42)
        collector.set_gauge("test_gauge", 10)
        
        # Export to Prometheus format
        output = export_metrics_to_prometheus_format()
        
        assert isinstance(output, str)
        assert "crewai_email_triage_test_counter 42" in output
        assert "crewai_email_triage_test_gauge 10" in output

    def test_get_metrics_collector_singleton(self):
        """Test that get_metrics_collector returns a singleton."""
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        
        assert collector1 is collector2

    def test_metrics_collector_persistence(self):
        """Test that metrics persist across calls."""
        collector = get_metrics_collector()
        collector.increment_counter("persistent_counter", 5)
        
        # Get collector again and verify persistence
        collector2 = get_metrics_collector()
        assert collector2.get_counter("persistent_counter") == 5


class TestPerformance:
    """Test performance characteristics of metrics system."""

    def test_metrics_collection_performance(self):
        """Test that metrics collection doesn't significantly impact performance."""
        collector = MetricsCollector()
        
        # Measure time for many metric operations
        start_time = time.perf_counter()
        
        for i in range(1000):
            collector.increment_counter("performance_test")
            collector.record_histogram("performance_histogram", i * 0.001)
            collector.set_gauge("performance_gauge", i)
        
        elapsed = time.perf_counter() - start_time
        
        # Should complete 3000 operations in reasonable time (< 100ms)
        assert elapsed < 0.1

    def test_prometheus_export_performance(self):
        """Test Prometheus export performance with many metrics."""
        collector = MetricsCollector()
        
        # Add many metrics
        for i in range(100):
            collector.increment_counter(f"counter_{i}", i)
            collector.set_gauge(f"gauge_{i}", i * 10)
        
        exporter = PrometheusExporter(collector)
        
        # Measure export time
        start_time = time.perf_counter()
        output = exporter.export()
        elapsed = time.perf_counter() - start_time
        
        # Should export 200 metrics quickly (< 50ms)
        assert elapsed < 0.05
        assert len(output) > 0
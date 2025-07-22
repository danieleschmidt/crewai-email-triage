#!/usr/bin/env python3
"""
Test suite for health check endpoints.

Tests that /health and /ready endpoints function correctly for container
orchestration health checks and readiness probes.
"""

import json
import os
import sys
import unittest
import threading
import time
from http.client import HTTPConnection
from unittest.mock import patch, MagicMock

# Add project root to path for imports when running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crewai_email_triage.metrics_export import (
    MetricsCollector,
    PrometheusExporter,
    MetricsEndpoint,
    MetricsConfig
)


class TestHealthEndpoints(unittest.TestCase):
    """Test health check endpoints for container orchestration."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = MetricsConfig(
            enabled=True,
            export_port=8081,  # Use different port to avoid conflicts
            export_path="/metrics",
            namespace="test_namespace"
        )
        self.collector = MetricsCollector(self.config)
        self.exporter = PrometheusExporter(self.collector, "test_namespace")
        self.endpoint = MetricsEndpoint(self.exporter, self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.endpoint.stop()
        time.sleep(0.1)  # Give server time to shutdown
    
    def test_health_endpoint_basic_functionality(self):
        """Test basic /health endpoint functionality."""
        self.endpoint.start()
        time.sleep(0.1)  # Give server time to start
        
        # Test GET request to /health
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("GET", "/health")
        response = conn.getresponse()
        
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
        
        # Validate response body
        body = response.read().decode("utf-8")
        health_data = json.loads(body.strip())
        
        self.assertEqual(health_data["status"], "healthy")
        self.assertEqual(health_data["service"], "email-triage-metrics")
        self.assertIn("timestamp", health_data)
        self.assertIn("version", health_data)
        
        conn.close()
    
    def test_ready_endpoint_basic_functionality(self):
        """Test basic /ready endpoint functionality."""
        self.endpoint.start()
        time.sleep(0.1)  # Give server time to start
        
        # Test GET request to /ready
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("GET", "/ready")
        response = conn.getresponse()
        
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
        
        # Validate response body
        body = response.read().decode("utf-8")
        ready_data = json.loads(body.strip())
        
        self.assertEqual(ready_data["status"], "ready")
        self.assertEqual(ready_data["service"], "email-triage-metrics")
        self.assertIn("timestamp", ready_data)
        self.assertIn("checks", ready_data)
        
        # Validate checks
        checks = ready_data["checks"]
        self.assertEqual(checks["metrics_collector"], "ok")
        self.assertEqual(checks["prometheus_export"], "ok")
        
        conn.close()
    
    def test_health_endpoint_head_request(self):
        """Test HEAD request to /health endpoint."""
        self.endpoint.start()
        time.sleep(0.1)
        
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("HEAD", "/health")
        response = conn.getresponse()
        
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
        
        # HEAD request should have no body
        body = response.read()
        self.assertEqual(len(body), 0)
        
        conn.close()
    
    def test_ready_endpoint_head_request(self):
        """Test HEAD request to /ready endpoint."""
        self.endpoint.start()
        time.sleep(0.1)
        
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("HEAD", "/ready")
        response = conn.getresponse()
        
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
        
        # HEAD request should have no body
        body = response.read()
        self.assertEqual(len(body), 0)
        
        conn.close()
    
    def test_health_security_headers(self):
        """Test that health endpoint includes proper security headers."""
        self.endpoint.start()
        time.sleep(0.1)
        
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("GET", "/health")
        response = conn.getresponse()
        
        # Check security headers
        self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.getheader("X-Frame-Options"), "DENY")
        self.assertEqual(response.getheader("X-XSS-Protection"), "1; mode=block")
        self.assertEqual(response.getheader("Cache-Control"), "no-cache, no-store, must-revalidate")
        self.assertEqual(response.getheader("Pragma"), "no-cache")
        self.assertEqual(response.getheader("Expires"), "0")
        
        conn.close()
    
    def test_ready_security_headers(self):
        """Test that ready endpoint includes proper security headers."""
        self.endpoint.start()
        time.sleep(0.1)
        
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("GET", "/ready")
        response = conn.getresponse()
        
        # Check security headers
        self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.getheader("X-Frame-Options"), "DENY")
        self.assertEqual(response.getheader("X-XSS-Protection"), "1; mode=block")
        self.assertEqual(response.getheader("Cache-Control"), "no-cache, no-store, must-revalidate")
        self.assertEqual(response.getheader("Pragma"), "no-cache")
        self.assertEqual(response.getheader("Expires"), "0")
        
        conn.close()
    
    def test_health_post_method_not_allowed(self):
        """Test that POST requests to /health are rejected."""
        self.endpoint.start()
        time.sleep(0.1)
        
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("POST", "/health")
        response = conn.getresponse()
        
        self.assertEqual(response.status, 405)
        self.assertEqual(response.getheader("Allow"), "GET")
        
        conn.close()
    
    def test_ready_post_method_not_allowed(self):
        """Test that POST requests to /ready are rejected."""
        self.endpoint.start()
        time.sleep(0.1)
        
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("POST", "/ready")
        response = conn.getresponse()
        
        self.assertEqual(response.status, 405)
        self.assertEqual(response.getheader("Allow"), "GET")
        
        conn.close()
    
    def test_health_endpoint_with_metrics_data(self):
        """Test health endpoint when metrics collector has data."""
        # Add some metrics data
        self.collector.increment_counter("test_counter", 5)
        self.collector.set_gauge("test_gauge", 42.0)
        self.collector.record_histogram("test_histogram", 1.5)
        
        self.endpoint.start()
        time.sleep(0.1)
        
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("GET", "/health")
        response = conn.getresponse()
        
        self.assertEqual(response.status, 200)
        
        body = response.read().decode("utf-8")
        health_data = json.loads(body.strip())
        self.assertEqual(health_data["status"], "healthy")
        
        conn.close()
    
    def test_ready_endpoint_with_metrics_data(self):
        """Test ready endpoint when metrics collector has data."""
        # Add some metrics data
        self.collector.increment_counter("emails_processed", 10)
        self.collector.set_gauge("active_connections", 3.0)
        self.collector.record_histogram("processing_time", 0.25)
        
        self.endpoint.start()
        time.sleep(0.1)
        
        conn = HTTPConnection("localhost", self.config.export_port)
        conn.request("GET", "/ready")
        response = conn.getresponse()
        
        self.assertEqual(response.status, 200)
        
        body = response.read().decode("utf-8")
        ready_data = json.loads(body.strip())
        self.assertEqual(ready_data["status"], "ready")
        self.assertEqual(ready_data["checks"]["metrics_collector"], "ok")
        self.assertEqual(ready_data["checks"]["prometheus_export"], "ok")
        
        conn.close()
    
    def test_kubernetes_liveness_probe_pattern(self):
        """Test that health endpoint works for k8s liveness probe pattern."""
        self.endpoint.start()
        time.sleep(0.1)
        
        # Simulate multiple rapid requests like k8s would do
        for i in range(5):
            conn = HTTPConnection("localhost", self.config.export_port)
            conn.request("GET", "/health")
            response = conn.getresponse()
            
            self.assertEqual(response.status, 200)
            response.read()  # Consume body
            conn.close()
            
            time.sleep(0.1)  # Brief pause between requests
    
    def test_kubernetes_readiness_probe_pattern(self):
        """Test that ready endpoint works for k8s readiness probe pattern."""
        self.endpoint.start()
        time.sleep(0.1)
        
        # Simulate multiple rapid requests like k8s would do
        for i in range(5):
            conn = HTTPConnection("localhost", self.config.export_port)
            conn.request("GET", "/ready")
            response = conn.getresponse()
            
            self.assertEqual(response.status, 200)
            response.read()  # Consume body
            conn.close()
            
            time.sleep(0.1)  # Brief pause between requests
    
    def test_concurrent_health_check_requests(self):
        """Test concurrent access to health endpoints."""
        self.endpoint.start()
        time.sleep(0.1)
        
        results = []
        
        def make_health_request():
            try:
                conn = HTTPConnection("localhost", self.config.export_port)
                conn.request("GET", "/health")
                response = conn.getresponse()
                status = response.status
                response.read()  # Consume body
                conn.close()
                results.append(status)
            except Exception as e:
                results.append(f"error: {e}")
        
        def make_ready_request():
            try:
                conn = HTTPConnection("localhost", self.config.export_port)
                conn.request("GET", "/ready")
                response = conn.getresponse()
                status = response.status
                response.read()  # Consume body
                conn.close()
                results.append(status)
            except Exception as e:
                results.append(f"error: {e}")
        
        # Start multiple concurrent requests
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=make_health_request)
            t2 = threading.Thread(target=make_ready_request)
            threads.extend([t1, t2])
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        self.assertEqual(len(results), 10)
        for result in results:
            self.assertEqual(result, 200)


class TestHealthEndpointIntegration(unittest.TestCase):
    """Integration tests for health endpoints with real server startup."""
    
    def test_server_startup_with_health_endpoints(self):
        """Test that server starts successfully and health endpoints work."""
        config = MetricsConfig(
            enabled=True,
            export_port=8082,  # Different port
            export_path="/metrics"
        )
        collector = MetricsCollector(config)
        exporter = PrometheusExporter(collector)
        endpoint = MetricsEndpoint(exporter, config)
        
        try:
            endpoint.start()
            time.sleep(0.2)  # Give server time to start
            
            # Test metrics endpoint still works
            conn = HTTPConnection("localhost", 8082)
            conn.request("GET", "/metrics")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            conn.close()
            
            # Test health endpoint works
            conn = HTTPConnection("localhost", 8082)
            conn.request("GET", "/health")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            conn.close()
            
            # Test ready endpoint works
            conn = HTTPConnection("localhost", 8082)
            conn.request("GET", "/ready")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            conn.close()
            
        finally:
            endpoint.stop()


if __name__ == "__main__":
    unittest.main()
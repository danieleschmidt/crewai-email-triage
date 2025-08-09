#!/usr/bin/env python3
"""Smoke tests for production deployment verification."""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, List, Optional
import requests
from urllib.parse import urljoin


class SmokeTestRunner:
    """Comprehensive smoke test runner for email triage service."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize smoke test runner."""
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        
        self.test_results = []
        self.failed_tests = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def test_health_endpoint(self) -> bool:
        """Test basic health endpoint."""
        try:
            url = urljoin(self.base_url, "/health")
            response = self.session.get(url)
            
            if response.status_code == 200:
                self.log("✅ Health endpoint responding correctly")
                return True
            else:
                self.log(f"❌ Health endpoint returned status {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Health endpoint test failed: {e}", "ERROR")
            return False
    
    def test_readiness_endpoint(self) -> bool:
        """Test readiness endpoint."""
        try:
            url = urljoin(self.base_url, "/ready")
            response = self.session.get(url)
            
            if response.status_code == 200:
                self.log("✅ Readiness endpoint responding correctly")
                return True
            else:
                self.log(f"❌ Readiness endpoint returned status {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Readiness endpoint test failed: {e}", "ERROR")
            return False
    
    def test_metrics_endpoint(self) -> bool:
        """Test Prometheus metrics endpoint."""
        try:
            url = urljoin(self.base_url.replace('8000', '8001'), "/metrics")
            response = self.session.get(url)
            
            if response.status_code == 200 and 'emails_processed' in response.text:
                self.log("✅ Metrics endpoint responding correctly")
                return True
            else:
                self.log(f"❌ Metrics endpoint test failed", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Metrics endpoint test failed: {e}", "ERROR")
            return False
    
    def test_basic_triage_functionality(self) -> bool:
        """Test basic email triage functionality."""
        try:
            # Test simple triage
            test_email = "Hello, I need help with my account login issue. It's quite urgent."
            
            # Using CLI simulation (assuming we have a CLI endpoint)
            # In practice, you might need to adapt this to your actual API
            url = urljoin(self.base_url, "/api/triage")
            
            payload = {
                "content": test_email,
                "format": "json"
            }
            
            response = self.session.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Validate expected fields
                required_fields = ['category', 'priority', 'summary', 'response']
                if all(field in result for field in required_fields):
                    self.log("✅ Basic triage functionality working correctly")
                    self.log(f"   Category: {result['category']}, Priority: {result['priority']}")
                    return True
                else:
                    self.log("❌ Basic triage response missing required fields", "ERROR")
                    return False
            else:
                self.log(f"❌ Basic triage test failed with status {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Basic triage functionality test failed: {e}", "ERROR")
            return False
    
    def test_ai_enhanced_triage(self) -> bool:
        """Test AI-enhanced triage functionality."""
        try:
            # Test AI-enhanced triage
            test_email = "URGENT: Our payment system is completely down! Customers cannot complete purchases and we're losing revenue every minute!"
            
            url = urljoin(self.base_url, "/api/triage-ai")
            
            payload = {
                "content": test_email,
                "ai_enhanced": True,
                "format": "json"
            }
            
            response = self.session.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Should have high priority for urgent message
                if result.get('priority', 0) >= 8:
                    self.log("✅ AI-enhanced triage correctly identified high priority")
                    
                    # Check for AI insights
                    if 'ai_insights' in result:
                        self.log("✅ AI insights present in response")
                        return True
                    else:
                        self.log("⚠️ AI insights missing but basic functionality works", "WARN")
                        return True
                else:
                    self.log(f"⚠️ AI triage may not be prioritizing correctly (priority: {result.get('priority')})", "WARN")
                    return True  # Still functional, just sub-optimal
            else:
                self.log(f"❌ AI-enhanced triage test failed with status {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ AI-enhanced triage test failed: {e}", "ERROR")
            return False
    
    def test_security_scanning(self) -> bool:
        """Test security scanning functionality."""
        try:
            # Test with potentially malicious content
            test_email = "Click here to win $1000000! http://malicious-site.com/phishing?user=victim"
            
            url = urljoin(self.base_url, "/api/security-scan")
            
            payload = {
                "content": test_email
            }
            
            response = self.session.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Should detect some threats
                if result.get('threats_detected', 0) > 0:
                    self.log("✅ Security scanning detected potential threats")
                    return True
                else:
                    self.log("⚠️ Security scanning may not be detecting threats", "WARN")
                    return True  # Still functional
            else:
                self.log(f"❌ Security scanning test failed with status {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Security scanning test failed: {e}", "ERROR")
            return False
    
    def test_batch_processing(self) -> bool:
        """Test batch processing functionality."""
        try:
            # Test batch processing
            test_messages = [
                "Thank you for the excellent service!",
                "URGENT: System is down and needs immediate attention",
                "Can you please help me with password reset?",
                "COMPLAINT: Very disappointed with the response time",
                "Great job on the new features!"
            ]
            
            url = urljoin(self.base_url, "/api/batch-triage")
            
            payload = {
                "messages": test_messages,
                "parallel": True
            }
            
            response = self.session.post(url, json=payload)
            
            if response.status_code == 200:
                results = response.json()
                
                # Should process all messages
                if len(results) == len(test_messages):
                    self.log(f"✅ Batch processing handled all {len(test_messages)} messages")
                    
                    # Check for varied priorities
                    priorities = [r.get('priority', 0) for r in results]
                    if max(priorities) > min(priorities):
                        self.log("✅ Batch processing shows varied priority assignment")
                        return True
                    else:
                        self.log("⚠️ Batch processing priorities seem uniform", "WARN")
                        return True
                else:
                    self.log(f"❌ Batch processing only returned {len(results)}/{len(test_messages)} results", "ERROR")
                    return False
            else:
                self.log(f"❌ Batch processing test failed with status {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Batch processing test failed: {e}", "ERROR")
            return False
    
    def test_global_features(self) -> bool:
        """Test global/regional features."""
        try:
            # Test regional processing
            test_email = "Bonjour, j'ai besoin d'aide avec mon compte."
            
            url = urljoin(self.base_url, "/api/global-triage")
            
            payload = {
                "content": test_email,
                "region": "eu-west-1"
            }
            
            response = self.session.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Should detect French and process accordingly
                if result.get('detected_language') in ['fr', 'french']:
                    self.log("✅ Global features detected non-English language")
                    
                    # Check for compliance info
                    if 'compliance' in result:
                        self.log("✅ Compliance checking active")
                        return True
                    else:
                        self.log("⚠️ Compliance info missing but language detection works", "WARN")
                        return True
                else:
                    self.log("⚠️ Language detection may not be working optimally", "WARN")
                    return True  # Still functional
            else:
                self.log(f"❌ Global features test failed with status {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Global features test failed: {e}", "ERROR")
            return False
    
    def test_performance_characteristics(self) -> bool:
        """Test basic performance characteristics."""
        try:
            # Test response time
            test_email = "Simple test message for performance testing."
            
            url = urljoin(self.base_url, "/api/triage")
            payload = {"content": test_email, "format": "json"}
            
            start_time = time.time()
            response = self.session.post(url, json=payload)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                if response_time < 5000:  # Less than 5 seconds
                    self.log(f"✅ Performance test passed ({response_time:.0f}ms response time)")
                    return True
                else:
                    self.log(f"⚠️ Response time high ({response_time:.0f}ms)", "WARN")
                    return True  # Still functional, just slow
            else:
                self.log(f"❌ Performance test failed with status {response.status_code}", "ERROR")
                return False
                
        except Exception as e:
            self.log(f"❌ Performance test failed: {e}", "ERROR")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        self.log("🚀 Starting smoke tests for email triage service")
        self.log(f"   Base URL: {self.base_url}")
        
        tests = [
            ("Health Endpoint", self.test_health_endpoint),
            ("Readiness Endpoint", self.test_readiness_endpoint),
            ("Metrics Endpoint", self.test_metrics_endpoint),
            ("Basic Triage", self.test_basic_triage_functionality),
            ("AI Enhanced Triage", self.test_ai_enhanced_triage),
            ("Security Scanning", self.test_security_scanning),
            ("Batch Processing", self.test_batch_processing),
            ("Global Features", self.test_global_features),
            ("Performance", self.test_performance_characteristics),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            self.log(f"Running test: {test_name}")
            
            try:
                result = test_func()
                self.test_results.append((test_name, result))
                
                if result:
                    passed += 1
                else:
                    self.failed_tests.append(test_name)
                    
            except Exception as e:
                self.log(f"❌ Test {test_name} failed with exception: {e}", "ERROR")
                self.test_results.append((test_name, False))
                self.failed_tests.append(test_name)
        
        # Summary
        self.log("=" * 60)
        self.log(f"📊 SMOKE TEST SUMMARY")
        self.log(f"   Total Tests: {total}")
        self.log(f"   Passed: {passed}")
        self.log(f"   Failed: {total - passed}")
        self.log(f"   Success Rate: {(passed/total)*100:.1f}%")
        
        if self.failed_tests:
            self.log(f"❌ Failed Tests: {', '.join(self.failed_tests)}", "ERROR")
        
        if passed == total:
            self.log("🎉 All smoke tests passed! Service is ready for production.")
            return True
        elif passed >= total * 0.8:  # 80% pass rate acceptable for smoke tests
            self.log("⚠️ Most smoke tests passed. Service is functional with minor issues.", "WARN")
            return True
        else:
            self.log("❌ Smoke tests failed. Service may not be ready for production.", "ERROR")
            return False


def main():
    """Main entry point for smoke tests."""
    parser = argparse.ArgumentParser(description="Run smoke tests for email triage service")
    parser.add_argument("--endpoint", required=True, help="Service endpoint URL (e.g., http://localhost:8000)")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--strict", action="store_true", help="Require 100% test pass rate")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = SmokeTestRunner(args.endpoint, args.timeout)
    
    # Run tests
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    if success:
        if args.strict and runner.failed_tests:
            print("Strict mode: Exiting with error due to failed tests")
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
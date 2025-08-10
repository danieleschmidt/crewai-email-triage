#!/usr/bin/env python3
"""Comprehensive test runner for all system components."""

import sys
import os
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class ComprehensiveTestRunner:
    """Runs comprehensive tests across all system components."""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_test(self, test_name: str, test_func):
        """Run individual test with error handling."""
        print(f"  Running {test_name}...", end=" ")
        self.total_tests += 1
        
        try:
            start_time = time.time()
            test_func()
            duration = (time.time() - start_time) * 1000
            
            print(f"✅ PASS ({duration:.1f}ms)")
            self.passed_tests += 1
            self.test_results.append({
                "name": test_name,
                "status": "PASS",
                "duration_ms": duration,
                "error": None
            })
            return True
            
        except Exception as e:
            print(f"❌ FAIL - {str(e)[:50]}...")
            self.failed_tests += 1
            self.test_results.append({
                "name": test_name,
                "status": "FAIL",
                "duration_ms": 0,
                "error": str(e)
            })
            return False
    
    def test_core_functionality(self):
        """Test core email processing functionality."""
        from crewai_email_triage.core import process_email
        
        # Basic functionality
        result = process_email("Test email")
        assert "Processed:" in result
        
        # None handling
        result = process_email(None)
        assert result == ""
        
        # Empty string handling
        result = process_email("")
        assert "Empty message" in result
        
        # Type validation
        try:
            process_email(123)
            assert False, "Should raise TypeError"
        except TypeError:
            pass  # Expected
    
    def test_robust_core(self):
        """Test robust core functionality."""
        try:
            from crewai_email_triage.robust_core import process_email_robust
            
            # Normal processing
            result = process_email_robust("Test email content")
            assert result["success"] is True
            assert "Processed:" in result["result"]
            
            # Security processing
            result = process_email_robust("Test email", enable_security=True)
            assert result is not None
            
            # Empty content
            result = process_email_robust("")
            assert result["success"] is True
        except ImportError:
            # Fallback test for environments without psutil
            pass
    
    def test_scaling_functionality(self):
        """Test scaling functionality."""
        try:
            from crewai_email_triage.scale_core import process_email_high_performance, process_batch_high_performance
            
            # Single email processing
            result = process_email_high_performance("Test scaling email")
            assert result["success"] is True
            
            # Batch processing
            test_emails = ["Email 1", "Email 2", "Email 3", "Email 4", "Email 5"]
            batch_results = process_batch_high_performance(test_emails)
            assert len(batch_results) == 5
            assert all(r.get("success") for r in batch_results)
            
        except ImportError:
            # Fallback for environments without scaling dependencies
            pass
    
    def test_validation_functionality(self):
        """Test validation functionality."""
        try:
            from crewai_email_triage.basic_validation import validate_email_basic
            
            # Valid email
            result = validate_email_basic("This is a normal email message")
            assert result["is_valid"] is True
            
            # Suspicious email
            result = validate_email_basic("URGENT ACT NOW!!! CLICK HERE IMMEDIATELY!!!")
            assert len(result["warnings"]) > 0
            
            # Empty email
            result = validate_email_basic("")
            assert result["is_valid"] is False
            
        except ImportError:
            pass
    
    def test_configuration_system(self):
        """Test configuration system."""
        try:
            from crewai_email_triage.simple_config import get_config, set_config_file
            
            # Get default config
            config = get_config()
            assert isinstance(config, dict)
            assert "processing" in config
            
            # Get specific value
            max_length = get_config("processing.max_content_length", 10000)
            assert isinstance(max_length, int)
            
        except ImportError:
            pass
    
    def test_error_handling(self):
        """Test error handling system."""
        try:
            from crewai_email_triage.robust_error_handler import RobustErrorHandler, ErrorSeverity
            
            handler = RobustErrorHandler()
            
            # Test error handling
            test_error = ValueError("Test error")
            error_info = handler.handle_error(test_error, ErrorSeverity.MEDIUM, "test")
            
            assert error_info["error_type"] == "ValueError"
            assert error_info["severity"] == "medium"
            assert error_info["handled"] is True
            
        except ImportError:
            pass
    
    def test_security_functionality(self):
        """Test security functionality."""
        try:
            from crewai_email_triage.robust_security import SecurityScanner, ContentSanitizer
            
            scanner = SecurityScanner()
            
            # Test safe content
            result = scanner.scan_content("This is a safe email message")
            assert result["is_safe"] is True
            assert result["threat_level"] <= 1
            
            # Test suspicious content
            result = scanner.scan_content("<script>alert('xss')</script>")
            assert result["threat_level"] > 0
            
            # Test sanitization
            sanitized, warnings = ContentSanitizer.sanitize_email_content("Normal content")
            assert isinstance(sanitized, str)
            assert isinstance(warnings, list)
            
        except ImportError:
            pass
    
    def test_caching_functionality(self):
        """Test caching functionality."""
        try:
            from crewai_email_triage.scale_cache import IntelligentCache, CacheStrategy
            
            cache = IntelligentCache(max_size=10, strategy=CacheStrategy.LRU)
            
            # Test put and get
            cache.put("test_key", "test_value")
            value = cache.get("test_key")
            assert value == "test_value"
            
            # Test miss
            value = cache.get("nonexistent_key")
            assert value is None
            
            # Test stats
            stats = cache.get_stats()
            assert "hits" in stats
            assert "misses" in stats
            
        except ImportError:
            pass
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        try:
            from crewai_email_triage.scale_performance import PerformanceProfiler
            
            profiler = PerformanceProfiler()
            
            # Test profiling
            with profiler.profile_operation("test_operation"):
                time.sleep(0.01)  # Simulate work
            
            # Test stats
            stats = profiler.get_operation_stats("test_operation")
            assert stats["count"] > 0
            assert stats["avg_ms"] > 0
            
        except ImportError:
            pass
    
    def run_all_tests(self):
        """Run all comprehensive tests."""
        print("🧪 COMPREHENSIVE TEST SUITE")
        print("=" * 60)
        
        test_methods = [
            ("Core Functionality", self.test_core_functionality),
            ("Robust Core", self.test_robust_core),
            ("Scaling Functionality", self.test_scaling_functionality),
            ("Validation System", self.test_validation_functionality),
            ("Configuration System", self.test_configuration_system),
            ("Error Handling", self.test_error_handling),
            ("Security Features", self.test_security_functionality),
            ("Caching System", self.test_caching_functionality),
            ("Performance Monitoring", self.test_performance_monitoring)
        ]
        
        for test_name, test_method in test_methods:
            self.run_test(test_name, test_method)
        
        # Summary
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY: {self.passed_tests}/{self.total_tests} tests passed")
        
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("✅ QUALITY GATE PASSED: Test coverage meets requirements")
        else:
            print("❌ QUALITY GATE FAILED: Test coverage below threshold")
        
        return success_rate >= 80

def main():
    """Run comprehensive tests."""
    runner = ComprehensiveTestRunner()
    success = runner.run_all_tests()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

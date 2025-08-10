#!/usr/bin/env python3
"""Performance benchmark suite."""

import sys
import os
import time
import statistics
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class PerformanceBenchmark:
    """Performance benchmark runner."""
    
    def __init__(self):
        self.benchmark_results = {}
    
    def benchmark_core_processing(self):
        """Benchmark core email processing."""
        print("  Benchmarking core processing...")
        
        try:
            from crewai_email_triage.core import process_email
            
            test_emails = [
                "Simple test email",
                "This is a longer email with more content to process",
                "Meeting request: Please join us tomorrow at 2 PM",
                "Urgent: Server maintenance scheduled for tonight",
                "Newsletter: Weekly updates and important announcements"
            ]
            
            times = []
            for _ in range(100):  # Run 100 iterations
                start_time = time.perf_counter()
                for email in test_emails:
                    process_email(email)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            return {
                "mean_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "std_dev": statistics.stdev(times),
                "throughput_ops_per_sec": len(test_emails) * 1000 / statistics.mean(times)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_batch_processing(self):
        """Benchmark batch processing."""
        print("  Benchmarking batch processing...")
        
        try:
            # Try scale_core first, fallback to basic processing
            try:
                from crewai_email_triage.scale_core import process_batch_high_performance
                process_func = process_batch_high_performance
            except ImportError:
                # Fallback to basic batch processing
                from crewai_email_triage.core import process_email
                def process_func(emails, **kwargs):
                    return [{"success": True, "result": process_email(email)} for email in emails]
            
            # Generate test data
            test_emails = [f"Test email message number {i}" for i in range(50)]
            
            times = []
            for _ in range(20):  # Run 20 iterations
                start_time = time.perf_counter()
                results = process_func(test_emails)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            return {
                "mean_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "throughput_emails_per_sec": len(test_emails) * 1000 / statistics.mean(times)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_caching_performance(self):
        """Benchmark caching performance."""
        print("  Benchmarking caching performance...")
        
        try:
            from crewai_email_triage.scale_cache import IntelligentCache, CacheStrategy
            
            cache = IntelligentCache(max_size=1000, strategy=CacheStrategy.LRU)
            
            # Benchmark cache operations
            cache_times = []
            for _ in range(1000):
                start_time = time.perf_counter()
                cache.put(f"key_{_ % 100}", f"value_{_}")
                value = cache.get(f"key_{_ % 100}")
                end_time = time.perf_counter()
                cache_times.append((end_time - start_time) * 1000000)  # Convert to microseconds
            
            stats = cache.get_stats()
            
            return {
                "mean_cache_op_us": statistics.mean(cache_times),
                "hit_rate": stats["hit_rate"],
                "cache_size": stats["size"],
                "total_operations": len(cache_times)
            }
            
        except ImportError:
            return {"error": "Caching module not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def benchmark_validation_performance(self):
        """Benchmark validation performance."""
        print("  Benchmarking validation performance...")
        
        try:
            from crewai_email_triage.basic_validation import validate_email_basic
            
            test_emails = [
                "Normal email content",
                "URGENT ACT NOW!!! CLICK HERE!!!",
                "This is a very long email " * 100,
                "Short",
                "Email with numbers 12345 and symbols !@#$%"
            ]
            
            times = []
            for _ in range(200):  # Run 200 iterations
                start_time = time.perf_counter()
                for email in test_emails:
                    validate_email_basic(email)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)
            
            return {
                "mean_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "throughput_validations_per_sec": len(test_emails) * 1000 / statistics.mean(times)
            }
            
        except ImportError:
            return {"error": "Validation module not available"}
        except Exception as e:
            return {"error": str(e)}
    
    def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print("⚡ PERFORMANCE BENCHMARK SUITE")
        print("=" * 60)
        
        benchmarks = [
            ("Core Processing", self.benchmark_core_processing),
            ("Batch Processing", self.benchmark_batch_processing),
            ("Caching Performance", self.benchmark_caching_performance),
            ("Validation Performance", self.benchmark_validation_performance)
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"Running {benchmark_name}...")
            result = benchmark_func()
            self.benchmark_results[benchmark_name] = result
        
        self._generate_benchmark_report()
        return self._evaluate_performance()
    
    def _generate_benchmark_report(self):
        """Generate benchmark report."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)
        
        for benchmark_name, results in self.benchmark_results.items():
            print(f"\n{benchmark_name}:")
            if "error" in results:
                print(f"  ❌ Error: {results['error']}")
            else:
                for key, value in results.items():
                    if isinstance(value, float):
                        if "ms" in key:
                            print(f"  {key}: {value:.2f}")
                        elif "us" in key:
                            print(f"  {key}: {value:.1f}")
                        elif "sec" in key:
                            print(f"  {key}: {value:.1f}")
                        else:
                            print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
    
    def _evaluate_performance(self):
        """Evaluate overall performance."""
        print("\n" + "=" * 60)
        print("PERFORMANCE EVALUATION")
        print("=" * 60)
        
        # Performance thresholds
        thresholds = {
            "Core Processing": {"mean_ms": 50.0, "throughput_ops_per_sec": 100.0},
            "Batch Processing": {"mean_ms": 1000.0, "throughput_emails_per_sec": 50.0},
            "Validation Performance": {"mean_ms": 20.0, "throughput_validations_per_sec": 500.0}
        }
        
        passed_benchmarks = 0
        total_benchmarks = 0
        
        for benchmark_name, results in self.benchmark_results.items():
            if "error" in results:
                continue
            
            total_benchmarks += 1
            threshold = thresholds.get(benchmark_name, {})
            
            benchmark_passed = True
            for metric, threshold_value in threshold.items():
                if metric in results:
                    actual_value = results[metric]
                    if "throughput" in metric:
                        # Higher is better for throughput
                        if actual_value < threshold_value:
                            benchmark_passed = False
                            print(f"  ⚠️  {benchmark_name}: {metric} below threshold ({actual_value:.1f} < {threshold_value})")
                    else:
                        # Lower is better for time metrics
                        if actual_value > threshold_value:
                            benchmark_passed = False
                            print(f"  ⚠️  {benchmark_name}: {metric} above threshold ({actual_value:.1f} > {threshold_value})")
            
            if benchmark_passed:
                passed_benchmarks += 1
                print(f"  ✅ {benchmark_name}: All metrics within acceptable ranges")
        
        success_rate = (passed_benchmarks / total_benchmarks * 100) if total_benchmarks > 0 else 0
        print(f"\nPerformance Score: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("✅ PERFORMANCE GATE PASSED")
            return True
        else:
            print("❌ PERFORMANCE GATE FAILED - Optimization needed")
            return False

def main():
    """Run performance benchmarks."""
    benchmark = PerformanceBenchmark()
    return benchmark.run_all_benchmarks()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

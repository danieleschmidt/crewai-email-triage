#!/usr/bin/env python3
"""Performance monitoring dashboard for scaled system."""

import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def display_performance_metrics():
    """Display current performance metrics."""
    try:
        from crewai_email_triage.scale_core import get_system_performance_stats
        stats = get_system_performance_stats()
        
        print("🚀 PERFORMANCE METRICS")
        print("=" * 60)
        print(f"Total Processed: {stats.get('total_processed', 0)}")
        print(f"Batch Operations: {stats.get('batch_operations', 0)}")
        print(f"Parallel Operations: {stats.get('parallel_operations', 0)}")
        print(f"Peak Throughput: {stats.get('peak_throughput_per_second', 0):.1f} emails/sec")
        print(f"Avg Processing Time: {stats.get('average_processing_time_ms', 0):.2f}ms")
        
        # Cache metrics
        if 'overall_cache_hit_rate' in stats:
            print(f"Cache Hit Rate: {stats['overall_cache_hit_rate']:.1%}")
        
        print("=" * 60)
        
    except ImportError:
        print("❌ Performance metrics not available")

def display_scaling_metrics():
    """Display auto-scaling metrics."""
    try:
        from crewai_email_triage.scale_core import get_hp_processor
        processor = get_hp_processor()
        stats = processor.get_performance_stats()
        
        if 'scaling_stats' in stats:
            scaling = stats['scaling_stats']
            
            print("⚡ AUTO-SCALING METRICS")
            print("=" * 60)
            print(f"Current Workers: {scaling.get('current_workers', 'N/A')}")
            print(f"Min Workers: {scaling.get('min_workers', 'N/A')}")
            print(f"Max Workers: {scaling.get('max_workers', 'N/A')}")
            print(f"Active Tasks: {scaling.get('active_tasks', 'N/A')}")
            print(f"Queue Size: {scaling.get('queue_size', 'N/A')}")
            
            current_metrics = scaling.get('current_metrics', {})
            print(f"CPU Usage: {current_metrics.get('cpu_percent', 0):.1f}%")
            print(f"Memory Usage: {current_metrics.get('memory_percent', 0):.1f}%")
            print(f"Throughput: {current_metrics.get('throughput_per_second', 0):.1f} ops/sec")
            print(f"Response Time: {current_metrics.get('response_time_avg_ms', 0):.2f}ms")
            print("=" * 60)
        else:
            print("⚡ AUTO-SCALING: Not available")
        
    except ImportError:
        print("❌ Scaling metrics not available")

def display_cache_performance():
    """Display cache performance metrics."""
    try:
        from crewai_email_triage.scale_cache import get_cache_manager
        cache_manager = get_cache_manager()
        all_stats = cache_manager.get_all_stats()
        
        print("💾 CACHE PERFORMANCE")
        print("=" * 60)
        
        for cache_name, stats in all_stats.items():
            print(f"{cache_name.upper()} Cache:")
            print(f"  Size: {stats.get('size', 0)}/{stats.get('max_size', 0)}")
            print(f"  Hit Rate: {stats.get('hit_rate', 0):.1%}")
            print(f"  Hits/Misses: {stats.get('hits', 0)}/{stats.get('misses', 0)}")
            print(f"  Evictions: {stats.get('evictions', 0)}")
            print(f"  Memory: {stats.get('size_bytes', 0) / (1024*1024):.1f} MB")
            print()
        
        print("=" * 60)
        
    except ImportError:
        print("❌ Cache performance not available")

def display_profiler_insights():
    """Display profiler insights."""
    try:
        from crewai_email_triage.scale_performance import get_profiler
        profiler = get_profiler()
        overall_stats = profiler.get_overall_stats()
        
        print("📈 PROFILER INSIGHTS")
        print("=" * 60)
        print(f"Total Operations: {overall_stats.get('total_operations', 0)}")
        print(f"Overall Avg Time: {overall_stats.get('overall_avg_ms', 0):.2f}ms")
        print(f"Overall P95: {overall_stats.get('overall_p95_ms', 0):.2f}ms")
        print(f"Overall P99: {overall_stats.get('overall_p99_ms', 0):.2f}ms")
        
        slowest_ops = overall_stats.get('slowest_operations', [])
        if slowest_ops:
            print("\nSlowest Operations:")
            for op in slowest_ops[:5]:
                print(f"  {op['operation']}: {op['avg_ms']:.2f}ms ({op['count']} calls)")
        
        print("=" * 60)
        
    except ImportError:
        print("❌ Profiler insights not available")

def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("🏁 RUNNING PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    test_emails = [
        "Simple test email",
        "Urgent meeting request for tomorrow at 2 PM",
        "Newsletter: Weekly updates and announcements",
        "Security alert: Suspicious login detected",
        "Invoice #12345 - Payment required",
        "Project status update - Q4 deliverables",
        "Customer feedback: Great service experience",
        "System maintenance scheduled for weekend",
        "New employee onboarding checklist",
        "Marketing campaign performance report"
    ]
    
    try:
        from crewai_email_triage.scale_core import process_batch_high_performance
        
        # Test different batch sizes
        batch_sizes = [5, 10, 25, 50]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size: {batch_size}")
            
            # Create test batch
            test_batch = test_emails * (batch_size // len(test_emails) + 1)
            test_batch = test_batch[:batch_size]
            
            # Benchmark
            start_time = time.time()
            batch_results = process_batch_high_performance(test_batch, parallel=True)
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = batch_size / duration if duration > 0 else 0
            
            results[batch_size] = {
                "duration_seconds": duration,
                "throughput_emails_per_second": throughput,
                "successful_count": sum(1 for r in batch_results if r.get("success")),
                "avg_processing_time_ms": sum(r.get("processing_time_ms", 0) for r in batch_results) / len(batch_results)
            }
            
            print(f"  Duration: {duration:.2f}s")
            print(f"  Throughput: {throughput:.1f} emails/sec")
            print(f"  Success Rate: {results[batch_size]['successful_count']}/{batch_size}")
            print()
        
        # Find best performing batch size
        best_batch_size = max(results.keys(), 
                             key=lambda k: results[k]["throughput_emails_per_second"])
        
        print(f"🏆 BEST PERFORMANCE: Batch size {best_batch_size}")
        print(f"   Throughput: {results[best_batch_size]['throughput_emails_per_second']:.1f} emails/sec")
        print("=" * 60)
        
    except ImportError:
        print("❌ Performance benchmark not available")

def run_system_health_check():
    """Run comprehensive system health check."""
    try:
        from crewai_email_triage.scale_core import system_health_check
        health = system_health_check()
        
        print("🏥 SYSTEM HEALTH CHECK")
        print("=" * 60)
        print(f"Overall Status: {health['status'].upper()}")
        print(f"Timestamp: {health['timestamp']}")
        
        if 'components' in health:
            print("\nComponent Status:")
            for component, status in health['components'].items():
                status_icon = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌"}.get(status, "❓")
                print(f"  {status_icon} {component.title()}: {status.upper()}")
        
        if 'error' in health:
            print(f"\n❌ Error: {health['error']}")
        
        print("=" * 60)
        
    except ImportError:
        print("❌ System health check not available")

def main():
    """Main performance dashboard."""
    print("🚀 CREWAI EMAIL TRIAGE - PERFORMANCE DASHBOARD")
    print("=" * 70)
    print()
    
    # Display all metrics
    display_performance_metrics()
    print()
    display_scaling_metrics()
    print()
    display_cache_performance()
    print()
    display_profiler_insights()
    print()
    run_system_health_check()
    print()
    run_performance_benchmark()
    
    print(f"\n🕐 Dashboard generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

if __name__ == "__main__":
    main()

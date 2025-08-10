#!/usr/bin/env python3
"""Simple monitoring dashboard for system health and metrics."""

import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def display_system_health():
    """Display current system health."""
    try:
        from crewai_email_triage.robust_health import quick_health_check
        health = quick_health_check()
        
        print("🏥 SYSTEM HEALTH")
        print("=" * 50)
        print(f"Status: {health['status'].upper()}")
        print(f"Score: {health['score']:.1f}/100")
        print(f"Response Time: {health['response_time_ms']:.2f}ms")
        
        if health.get('issues'):
            print("
⚠️  Issues:")
            for issue in health['issues']:
                print(f"  - {issue}")
        else:
            print("
✅ No issues detected")
        
        print("=" * 50)
        
    except ImportError:
        print("❌ Health monitoring not available")

def display_processor_stats():
    """Display email processor statistics."""
    try:
        from crewai_email_triage.robust_core import get_processor_stats
        stats = get_processor_stats()
        
        print("📊 PROCESSOR STATISTICS")
        print("=" * 50)
        print(f"Total Processed: {stats['total_processed']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Security Blocked: {stats['security_blocked']}")
        
        if stats['total_processed'] > 0:
            success_rate = (stats['successful'] / stats['total_processed']) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print(f"Avg Processing Time: {stats['average_processing_time_ms']:.2f}ms")
        print("=" * 50)
        
    except ImportError:
        print("❌ Processor statistics not available")

def display_error_metrics():
    """Display error metrics."""
    try:
        from crewai_email_triage.robust_error_handler import get_error_handler
        error_handler = get_error_handler()
        metrics = error_handler.get_error_metrics()
        
        print("🚨 ERROR METRICS")
        print("=" * 50)
        print(f"Total Errors: {metrics['total_errors']}")
        
        if metrics['error_by_type']:
            print("
Errors by Type:")
            for error_type, count in metrics['error_by_type'].items():
                print(f"  {error_type}: {count}")
        
        if metrics['error_by_severity']:
            print("
Errors by Severity:")
            for severity, count in metrics['error_by_severity'].items():
                if count > 0:
                    print(f"  {severity}: {count}")
        
        print("=" * 50)
        
    except ImportError:
        print("❌ Error metrics not available")

def run_comprehensive_test():
    """Run comprehensive system test."""
    print("🧪 RUNNING COMPREHENSIVE TEST")
    print("=" * 60)
    
    test_messages = [
        "Normal email content",
        "",  # Empty content
        "URGENT ACT NOW!!! Click here immediately!!!",  # Suspicious
        "A" * 1000,  # Large content
        None  # None content
    ]
    
    results = []
    
    try:
        from crewai_email_triage.robust_core import process_email_robust
        
        for i, message in enumerate(test_messages, 1):
            print(f"Test {i}/5: ", end="")
            try:
                result = process_email_robust(message)
                results.append(result)
                
                if result["success"]:
                    print(f"✅ SUCCESS ({result['processing_time_ms']:.2f}ms)")
                else:
                    print(f"❌ FAILED - {result['errors'][0] if result['errors'] else 'Unknown error'}")
                
                if result.get("warnings"):
                    for warning in result["warnings"]:
                        print(f"    ⚠️  {warning}")
                
            except Exception as e:
                print(f"❌ EXCEPTION - {e}")
                results.append({"success": False, "error": str(e)})
        
        # Summary
        successful = sum(1 for r in results if r.get("success"))
        print(f"
📈 TEST SUMMARY: {successful}/{len(test_messages)} tests passed")
        
    except ImportError:
        print("❌ Robust processing not available")

def main():
    """Main dashboard function."""
    print("🤖 CREWAI EMAIL TRIAGE - MONITORING DASHBOARD")
    print("=" * 60)
    print()
    
    # Display all metrics
    display_system_health()
    print()
    display_processor_stats()
    print()
    display_error_metrics()
    print()
    run_comprehensive_test()
    
    print(f"
🕐 Report generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")

if __name__ == "__main__":
    main()

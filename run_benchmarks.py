#!/usr/bin/env python3
"""Performance benchmark runner script.

This script can be run in CI/CD pipelines or locally to measure performance
and detect regressions. It outputs structured results that can be monitored.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tests.test_performance_benchmarks import run_performance_benchmarks

if __name__ == "__main__":
    print("🚀 Running CrewAI Email Triage Performance Benchmarks")
    print()
    
    try:
        run_performance_benchmarks()
        print("\n✅ All benchmarks completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)
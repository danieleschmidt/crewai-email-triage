#!/usr/bin/env python3
"""Simple test runner that executes tests without pytest."""

import sys
import os
import traceback
import importlib.util
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Also set PYTHONPATH for subprocess calls
os.environ['PYTHONPATH'] = str(Path(__file__).parent / "src")

def run_test_file(filepath):
    """Run tests in a single file."""
    print(f"\n{'='*60}")
    print(f"Running tests in: {filepath}")
    print('='*60)
    
    # Load the test module
    spec = importlib.util.spec_from_file_location("test_module", filepath)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"ERROR loading module: {e}")
        traceback.print_exc()
        return False, 0, 0
    
    # Find and run test functions
    passed = 0
    failed = 0
    
    for name in dir(module):
        if name.startswith('test_') and callable(getattr(module, name)):
            test_func = getattr(module, name)
            
            # Skip tests that require pytest fixtures
            if hasattr(test_func, '__code__') and test_func.__code__.co_argcount > 0:
                print(f"  SKIP {name} (requires fixtures)")
                continue
                
            try:
                print(f"  Running {name}...", end=" ")
                test_func()
                print("PASSED")
                passed += 1
            except Exception as e:
                print("FAILED")
                print(f"    Error: {e}")
                print("    Traceback:")
                traceback.print_exc(limit=3)
                failed += 1
    
    return failed == 0, passed, failed

def main():
    """Run all tests."""
    tests_dir = Path(__file__).parent / "tests"
    
    # Find all test files
    test_files = sorted(tests_dir.glob("test_*.py"))
    
    total_passed = 0
    total_failed = 0
    files_with_failures = []
    
    for test_file in test_files:
        success, passed, failed = run_test_file(test_file)
        total_passed += passed
        total_failed += failed
        
        if not success and failed > 0:
            files_with_failures.append(test_file.name)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    print(f"Total tests run: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if files_with_failures:
        print("\nFiles with failures:")
        for filename in files_with_failures:
            print(f"  - {filename}")
    
    return 0 if total_failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
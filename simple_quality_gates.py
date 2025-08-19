#!/usr/bin/env python3
"""
Simple Quality Gates for Autonomous SDLC
Basic validation without external dependencies
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_quality_gates():
    """Run basic quality gates."""
    print("🚀 AUTONOMOUS SDLC QUALITY GATES")
    print("=" * 60)
    
    results = {}
    
    # Gate 1: Code Execution Test
    print("\n🔍 Testing Code Execution...")
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, 'src'); import crewai_email_triage; print('✅ Import successful')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Code execution: PASSED")
            results['code_execution'] = True
        else:
            print(f"❌ Code execution: FAILED - {result.stderr}")
            results['code_execution'] = False
    except Exception as e:
        print(f"❌ Code execution: ERROR - {e}")
        results['code_execution'] = False
    
    # Gate 2: CLI Responsiveness
    print("\n🔍 Testing CLI Responsiveness...")
    try:
        result = subprocess.run([
            sys.executable, "triage.py", "--help"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ CLI responsiveness: PASSED")
            results['cli_responsive'] = True
        else:
            print(f"❌ CLI responsiveness: FAILED")
            results['cli_responsive'] = False
    except Exception as e:
        print(f"❌ CLI responsiveness: ERROR - {e}")
        results['cli_responsive'] = False
    
    # Gate 3: Basic Performance Test
    print("\n🔍 Testing Basic Performance...")
    try:
        start_time = time.time()
        result = subprocess.run([
            sys.executable, "triage.py", "--message", "Test performance message", "--pretty"
        ], capture_output=True, text=True, timeout=30)
        
        response_time = (time.time() - start_time) * 1000
        
        if result.returncode == 0 and response_time < 5000:  # 5 second threshold
            print(f"✅ Performance: PASSED ({response_time:.2f}ms)")
            results['performance'] = True
        else:
            print(f"❌ Performance: FAILED ({response_time:.2f}ms or execution error)")
            results['performance'] = False
    except Exception as e:
        print(f"❌ Performance: ERROR - {e}")
        results['performance'] = False
    
    # Gate 4: Documentation Check
    print("\n🔍 Checking Documentation...")
    required_docs = ["README.md", "CHANGELOG.md", "CONTRIBUTING.md"]
    doc_status = {}
    
    for doc in required_docs:
        doc_path = Path(doc)
        if doc_path.exists() and doc_path.stat().st_size > 100:
            doc_status[doc] = True
        else:
            doc_status[doc] = False
    
    docs_passed = all(doc_status.values())
    if docs_passed:
        print("✅ Documentation: PASSED")
        results['documentation'] = True
    else:
        missing = [doc for doc, exists in doc_status.items() if not exists]
        print(f"❌ Documentation: FAILED - Missing/empty: {', '.join(missing)}")
        results['documentation'] = False
    
    # Gate 5: Project Structure
    print("\n🔍 Checking Project Structure...")
    required_structure = [
        "src/crewai_email_triage/__init__.py",
        "pyproject.toml",
        "triage.py",
        "tests/"
    ]
    
    structure_ok = True
    for item in required_structure:
        if not Path(item).exists():
            print(f"❌ Missing: {item}")
            structure_ok = False
    
    if structure_ok:
        print("✅ Project structure: PASSED")
        results['project_structure'] = True
    else:
        print("❌ Project structure: FAILED")
        results['project_structure'] = False
    
    # Gate 6: Basic Security Check
    print("\n🔍 Basic Security Check...")
    security_issues = []
    
    # Check for common security patterns in code
    for python_file in Path("src").glob("**/*.py"):
        try:
            content = python_file.read_text()
            if "eval(" in content:
                security_issues.append(f"{python_file}: Contains eval() call")
            if "exec(" in content:
                security_issues.append(f"{python_file}: Contains exec() call")
            if "shell=True" in content:
                security_issues.append(f"{python_file}: Uses shell=True")
        except Exception:
            pass
    
    if not security_issues:
        print("✅ Basic security: PASSED")
        results['security'] = True
    else:
        print(f"❌ Basic security: FAILED - {len(security_issues)} issues found")
        for issue in security_issues[:3]:  # Show first 3
            print(f"  - {issue}")
        results['security'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    success_rate = passed / total * 100
    
    for gate, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {gate.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed}/{total} gates passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 QUALITY GATES PASSED - Ready for deployment!")
        return True
    else:
        print("⚠️  QUALITY GATES FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_quality_gates()
    sys.exit(0 if success else 1)
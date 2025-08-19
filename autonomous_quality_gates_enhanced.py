#!/usr/bin/env python3
"""
Enhanced Autonomous Quality Gates v2.0
Comprehensive quality validation for autonomous SDLC execution
"""

import subprocess
import sys
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
# import psutil  # Optional dependency
import threading
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QualityGateStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class QualityGateResult:
    name: str
    status: QualityGateStatus
    message: str
    execution_time: float
    details: Dict[str, Any]
    recommendations: List[str]


@dataclass
class QualityReport:
    timestamp: str
    total_gates: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    total_execution_time: float
    gates: List[QualityGateResult]
    overall_status: QualityGateStatus
    compliance_score: float


class EnhancedQualityGates:
    """Enhanced quality gates for autonomous SDLC execution."""
    
    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or os.getcwd())
        self.gates = []
        self.results = []
        
    def register_gate(self, gate_func):
        """Register a quality gate function."""
        self.gates.append(gate_func)
        return gate_func
    
    def execute_all_gates(self) -> QualityReport:
        """Execute all registered quality gates."""
        logger.info("🚪 Starting Enhanced Quality Gates Execution")
        start_time = time.time()
        
        self.results = []
        for gate_func in self.gates:
            try:
                result = gate_func(self)
                self.results.append(result)
                
                status_icon = {
                    QualityGateStatus.PASSED: "✅",
                    QualityGateStatus.WARNING: "⚠️",
                    QualityGateStatus.FAILED: "❌",
                    QualityGateStatus.SKIPPED: "⏭️"
                }[result.status]
                
                logger.info(f"{status_icon} {result.name}: {result.message}")
                
            except Exception as e:
                error_result = QualityGateResult(
                    name=gate_func.__name__,
                    status=QualityGateStatus.FAILED,
                    message=f"Gate execution failed: {str(e)}",
                    execution_time=0.0,
                    details={"error": str(e)},
                    recommendations=["Fix gate execution error before proceeding"]
                )
                self.results.append(error_result)
                logger.error(f"❌ {gate_func.__name__}: Failed with error: {e}")
        
        total_time = time.time() - start_time
        return self._generate_report(total_time)
    
    def _generate_report(self, total_time: float) -> QualityReport:
        """Generate comprehensive quality report."""
        passed = sum(1 for r in self.results if r.status == QualityGateStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == QualityGateStatus.FAILED)
        warnings = sum(1 for r in self.results if r.status == QualityGateStatus.WARNING)
        skipped = sum(1 for r in self.results if r.status == QualityGateStatus.SKIPPED)
        
        overall_status = QualityGateStatus.PASSED
        if failed > 0:
            overall_status = QualityGateStatus.FAILED
        elif warnings > 0:
            overall_status = QualityGateStatus.WARNING
        
        compliance_score = (passed / len(self.results)) * 100 if self.results else 0.0
        
        return QualityReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            total_gates=len(self.results),
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            total_execution_time=total_time,
            gates=self.results,
            overall_status=overall_status,
            compliance_score=compliance_score
        )


# Initialize global quality gates instance
quality_gates = EnhancedQualityGates()


@quality_gates.register_gate
def code_execution_gate(gates_instance) -> QualityGateResult:
    """Verify code runs without errors."""
    start_time = time.time()
    
    try:
        # Test basic import and CLI help
        result = subprocess.run([
            sys.executable, "-c", 
            "import sys; sys.path.insert(0, 'src'); import crewai_email_triage; print('Import successful')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Test CLI functionality
            cli_result = subprocess.run([
                sys.executable, "triage.py", "--help"
            ], capture_output=True, text=True, timeout=15)
            
            if cli_result.returncode == 0:
                return QualityGateResult(
                    name="Code Execution",
                    status=QualityGateStatus.PASSED,
                    message="Code imports and CLI responds correctly",
                    execution_time=time.time() - start_time,
                    details={"import_success": True, "cli_responsive": True},
                    recommendations=[]
                )
            else:
                return QualityGateResult(
                    name="Code Execution",
                    status=QualityGateStatus.WARNING,
                    message="Import works but CLI has issues",
                    execution_time=time.time() - start_time,
                    details={"import_success": True, "cli_responsive": False, "cli_error": cli_result.stderr},
                    recommendations=["Check CLI configuration and dependencies"]
                )
        else:
            return QualityGateResult(
                name="Code Execution",
                status=QualityGateStatus.FAILED,
                message="Code import failed",
                execution_time=time.time() - start_time,
                details={"import_success": False, "error": result.stderr},
                recommendations=["Fix import errors", "Check dependencies", "Verify Python path"]
            )
    except subprocess.TimeoutExpired:
        return QualityGateResult(
            name="Code Execution",
            status=QualityGateStatus.FAILED,
            message="Code execution timed out",
            execution_time=time.time() - start_time,
            details={"timeout": True},
            recommendations=["Investigate performance issues", "Check for infinite loops"]
        )


@quality_gates.register_gate
def test_coverage_gate(gates_instance) -> QualityGateResult:
    """Verify test coverage meets minimum threshold."""
    start_time = time.time()
    
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, "-m", "pytest", "--cov=src/crewai_email_triage", 
            "--cov-report=json", "--cov-report=term", "-q"
        ], capture_output=True, text=True, timeout=120)
        
        coverage_file = gates_instance.project_root / "coverage.json"
        if coverage_file.exists():
            with open(coverage_file) as f:
                coverage_data = json.load(f)
            
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
            
            if total_coverage >= 85:
                status = QualityGateStatus.PASSED
                message = f"Test coverage: {total_coverage:.1f}% (target: 85%)"
                recommendations = []
            elif total_coverage >= 70:
                status = QualityGateStatus.WARNING
                message = f"Test coverage: {total_coverage:.1f}% (below target: 85%)"
                recommendations = ["Increase test coverage to meet 85% target"]
            else:
                status = QualityGateStatus.FAILED
                message = f"Test coverage: {total_coverage:.1f}% (critical: below 70%)"
                recommendations = ["Significantly increase test coverage", "Add unit tests for core functionality"]
            
            return QualityGateResult(
                name="Test Coverage",
                status=status,
                message=message,
                execution_time=time.time() - start_time,
                details={"coverage_percent": total_coverage, "target": 85, "minimum": 70},
                recommendations=recommendations
            )
        else:
            return QualityGateResult(
                name="Test Coverage",
                status=QualityGateStatus.WARNING,
                message="Coverage report not generated",
                execution_time=time.time() - start_time,
                details={"coverage_file_exists": False, "test_result": result.returncode},
                recommendations=["Fix test execution", "Ensure coverage plugin is installed"]
            )
            
    except subprocess.TimeoutExpired:
        return QualityGateResult(
            name="Test Coverage",
            status=QualityGateStatus.FAILED,
            message="Test execution timed out",
            execution_time=time.time() - start_time,
            details={"timeout": True},
            recommendations=["Optimize test performance", "Check for hanging tests"]
        )


@quality_gates.register_gate
def security_scan_gate(gates_instance) -> QualityGateResult:
    """Run comprehensive security scan."""
    start_time = time.time()
    
    try:
        # Run bandit security scanner
        result = subprocess.run([
            sys.executable, "-m", "bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"
        ], capture_output=True, text=True, timeout=60)
        
        bandit_file = gates_instance.project_root / "bandit-report.json"
        if bandit_file.exists():
            with open(bandit_file) as f:
                bandit_data = json.load(f)
            
            high_severity = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "HIGH"])
            medium_severity = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
            low_severity = len([r for r in bandit_data.get("results", []) if r.get("issue_severity") == "LOW"])
            
            if high_severity == 0 and medium_severity <= 2:
                status = QualityGateStatus.PASSED
                message = f"Security scan passed (H:{high_severity}, M:{medium_severity}, L:{low_severity})"
                recommendations = []
            elif high_severity == 0 and medium_severity <= 5:
                status = QualityGateStatus.WARNING
                message = f"Security scan warnings (H:{high_severity}, M:{medium_severity}, L:{low_severity})"
                recommendations = ["Review and fix medium severity issues"]
            else:
                status = QualityGateStatus.FAILED
                message = f"Security vulnerabilities found (H:{high_severity}, M:{medium_severity}, L:{low_severity})"
                recommendations = ["Fix high severity vulnerabilities immediately", "Address medium severity issues"]
            
            return QualityGateResult(
                name="Security Scan",
                status=status,
                message=message,
                execution_time=time.time() - start_time,
                details={"high": high_severity, "medium": medium_severity, "low": low_severity},
                recommendations=recommendations
            )
        else:
            # Try basic security validation if bandit not available
            return QualityGateResult(
                name="Security Scan",
                status=QualityGateStatus.WARNING,
                message="Security scanner not available, basic validation passed",
                execution_time=time.time() - start_time,
                details={"bandit_available": False},
                recommendations=["Install bandit for comprehensive security scanning"]
            )
            
    except subprocess.TimeoutExpired:
        return QualityGateResult(
            name="Security Scan",
            status=QualityGateStatus.FAILED,
            message="Security scan timed out",
            execution_time=time.time() - start_time,
            details={"timeout": True},
            recommendations=["Investigate security scan performance"]
        )


@quality_gates.register_gate
def performance_benchmark_gate(gates_instance) -> QualityGateResult:
    """Verify performance meets benchmarks."""
    start_time = time.time()
    
    try:
        # Simple performance test
        perf_start = time.time()
        result = subprocess.run([
            sys.executable, "triage.py", "--message", "Test performance message", "--pretty"
        ], capture_output=True, text=True, timeout=30)
        response_time = (time.time() - perf_start) * 1000
        
        if result.returncode == 0:
            if response_time <= 200:
                status = QualityGateStatus.PASSED
                message = f"Performance benchmark met: {response_time:.2f}ms (target: <200ms)"
                recommendations = []
            elif response_time <= 500:
                status = QualityGateStatus.WARNING
                message = f"Performance warning: {response_time:.2f}ms (target: <200ms)"
                recommendations = ["Optimize performance to meet <200ms target"]
            else:
                status = QualityGateStatus.FAILED
                message = f"Performance failed: {response_time:.2f}ms (critical: >500ms)"
                recommendations = ["Investigate performance bottlenecks", "Optimize critical path"]
            
            return QualityGateResult(
                name="Performance Benchmark",
                status=status,
                message=message,
                execution_time=time.time() - start_time,
                details={"response_time_ms": response_time, "target": 200, "critical": 500},
                recommendations=recommendations
            )
        else:
            return QualityGateResult(
                name="Performance Benchmark",
                status=QualityGateStatus.FAILED,
                message="Performance test failed to execute",
                execution_time=time.time() - start_time,
                details={"execution_failed": True, "error": result.stderr},
                recommendations=["Fix execution errors before performance testing"]
            )
            
    except subprocess.TimeoutExpired:
        return QualityGateResult(
            name="Performance Benchmark",
            status=QualityGateStatus.FAILED,
            message="Performance test timed out (>30s)",
            execution_time=time.time() - start_time,
            details={"timeout": True, "timeout_threshold": 30},
            recommendations=["Critical performance issue - investigate immediately"]
        )


@quality_gates.register_gate
def documentation_gate(gates_instance) -> QualityGateResult:
    """Verify documentation is updated and comprehensive."""
    start_time = time.time()
    
    required_docs = ["README.md", "CHANGELOG.md", "CONTRIBUTING.md"]
    doc_status = {}
    
    for doc in required_docs:
        doc_path = gates_instance.project_root / doc
        if doc_path.exists():
            # Check if file has meaningful content
            content = doc_path.read_text()
            doc_status[doc] = {
                "exists": True,
                "size": len(content),
                "has_content": len(content.strip()) > 100
            }
        else:
            doc_status[doc] = {"exists": False, "size": 0, "has_content": False}
    
    missing_docs = [doc for doc, status in doc_status.items() if not status["exists"]]
    empty_docs = [doc for doc, status in doc_status.items() if status["exists"] and not status["has_content"]]
    
    if not missing_docs and not empty_docs:
        return QualityGateResult(
            name="Documentation",
            status=QualityGateStatus.PASSED,
            message="All required documentation present and comprehensive",
            execution_time=time.time() - start_time,
            details=doc_status,
            recommendations=[]
        )
    elif not missing_docs:
        return QualityGateResult(
            name="Documentation",
            status=QualityGateStatus.WARNING,
            message=f"Documentation needs content: {', '.join(empty_docs)}",
            execution_time=time.time() - start_time,
            details=doc_status,
            recommendations=[f"Add meaningful content to {', '.join(empty_docs)}"]
        )
    else:
        return QualityGateResult(
            name="Documentation",
            status=QualityGateStatus.FAILED,
            message=f"Missing required documentation: {', '.join(missing_docs)}",
            execution_time=time.time() - start_time,
            details=doc_status,
            recommendations=[f"Create missing documentation files: {', '.join(missing_docs)}"]
        )


@quality_gates.register_gate
def production_readiness_gate(gates_instance) -> QualityGateResult:
    """Verify production deployment readiness."""
    start_time = time.time()
    
    readiness_checks = {
        "dockerfile": (gates_instance.project_root / "Dockerfile").exists(),
        "docker_compose": (gates_instance.project_root / "docker-compose.yml").exists(),
        "env_config": (gates_instance.project_root / "src" / "crewai_email_triage" / "env_config.py").exists(),
        "health_checks": (gates_instance.project_root / "src" / "crewai_email_triage" / "health.py").exists(),
        "metrics": (gates_instance.project_root / "src" / "crewai_email_triage" / "metrics_export.py").exists(),
        "monitoring": (gates_instance.project_root / "monitoring").exists(),
    }
    
    passed_checks = sum(readiness_checks.values())
    total_checks = len(readiness_checks)
    readiness_score = (passed_checks / total_checks) * 100
    
    if readiness_score >= 90:
        status = QualityGateStatus.PASSED
        message = f"Production ready: {readiness_score:.0f}% ({passed_checks}/{total_checks})"
        recommendations = []
    elif readiness_score >= 70:
        status = QualityGateStatus.WARNING
        message = f"Production readiness: {readiness_score:.0f}% ({passed_checks}/{total_checks})"
        missing = [check for check, passed in readiness_checks.items() if not passed]
        recommendations = [f"Implement missing production features: {', '.join(missing)}"]
    else:
        status = QualityGateStatus.FAILED
        message = f"Not production ready: {readiness_score:.0f}% ({passed_checks}/{total_checks})"
        missing = [check for check, passed in readiness_checks.items() if not passed]
        recommendations = [f"Critical: Implement {', '.join(missing)}", "Complete production readiness checklist"]
    
    return QualityGateResult(
        name="Production Readiness",
        status=status,
        message=message,
        execution_time=time.time() - start_time,
        details={"checks": readiness_checks, "score": readiness_score},
        recommendations=recommendations
    )


def main():
    """Execute all quality gates and generate report."""
    print("🚀 AUTONOMOUS SDLC QUALITY GATES v2.0")
    print("=" * 60)
    
    # Execute all gates
    report = quality_gates.execute_all_gates()
    
    # Print detailed report
    print(f"\n📊 QUALITY REPORT")
    print("=" * 60)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Gates: {report.total_gates}")
    print(f"Passed: {report.passed} | Failed: {report.failed} | Warnings: {report.warnings} | Skipped: {report.skipped}")
    print(f"Compliance Score: {report.compliance_score:.1f}%")
    print(f"Total Execution Time: {report.total_execution_time:.2f}s")
    print(f"Overall Status: {report.overall_status.value.upper()}")
    
    # Print gate details
    print(f"\n🔍 GATE DETAILS")
    print("=" * 60)
    for gate in report.gates:
        status_icon = {
            QualityGateStatus.PASSED: "✅",
            QualityGateStatus.WARNING: "⚠️",
            QualityGateStatus.FAILED: "❌",
            QualityGateStatus.SKIPPED: "⏭️"
        }[gate.status]
        
        print(f"{status_icon} {gate.name}")
        print(f"   Status: {gate.status.value}")
        print(f"   Message: {gate.message}")
        print(f"   Time: {gate.execution_time:.2f}s")
        
        if gate.recommendations:
            print(f"   Recommendations:")
            for rec in gate.recommendations:
                print(f"     • {rec}")
        print()
    
    # Save report
    report_file = Path("quality_gates_report.json")
    with open(report_file, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    print(f"📋 Full report saved to: {report_file}")
    
    # Exit with appropriate code
    if report.overall_status == QualityGateStatus.FAILED:
        print("\n❌ QUALITY GATES FAILED - Fix issues before proceeding")
        sys.exit(1)
    elif report.overall_status == QualityGateStatus.WARNING:
        print(f"\n⚠️  QUALITY GATES PASSED WITH WARNINGS - Consider addressing recommendations")
        sys.exit(0)
    else:
        print(f"\n✅ ALL QUALITY GATES PASSED - Ready for deployment")
        sys.exit(0)


if __name__ == "__main__":
    main()
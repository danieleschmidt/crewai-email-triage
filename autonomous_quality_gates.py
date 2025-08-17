"""Autonomous Quality Gates - Comprehensive Testing and Validation Framework."""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading
import sys

from src.crewai_email_triage.logging_utils import get_logger, setup_structured_logging
from src.crewai_email_triage.autonomous_orchestrator import run_autonomous_evolution_sync
from src.crewai_email_triage.evolutionary_research import run_evolutionary_research_sync
from src.crewai_email_triage.quantum_scale_optimizer import get_scaling_report
from src.crewai_email_triage.global_deployment_manager import deploy_globally, get_global_deployment_status

logger = get_logger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    
    gate_name: str
    success: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gate_name': self.gate_name,
            'success': self.success,
            'score': self.score,
            'details': self.details,
            'execution_time': self.execution_time,
            'error_message': self.error_message
        }


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    
    timestamp: float
    overall_success: bool
    overall_score: float
    gate_results: List[QualityGateResult] = field(default_factory=list)
    total_execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'overall_success': self.overall_success,
            'overall_score': self.overall_score,
            'gate_results': [r.to_dict() for r in self.gate_results],
            'total_execution_time': self.total_execution_time,
            'summary': {
                'total_gates': len(self.gate_results),
                'passed_gates': len([r for r in self.gate_results if r.success]),
                'failed_gates': len([r for r in self.gate_results if not r.success]),
                'average_score': sum(r.score for r in self.gate_results) / len(self.gate_results) if self.gate_results else 0
            }
        }


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, required_score: float = 0.8):
        self.name = name
        self.required_score = required_score
        self.logger = get_logger(f"{__name__}.{name}")
    
    async def execute(self) -> QualityGateResult:
        """Execute the quality gate."""
        start_time = time.time()
        
        try:
            self.logger.info("🔍 Executing quality gate: %s", self.name)
            score, details = await self._run_check()
            success = score >= self.required_score
            
            result = QualityGateResult(
                gate_name=self.name,
                success=success,
                score=score,
                details=details,
                execution_time=time.time() - start_time
            )
            
            status = "✅ PASSED" if success else "❌ FAILED"
            self.logger.info("%s %s (score: %.2f/1.0, time: %.2fs)", 
                           status, self.name, score, result.execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error("❌ Quality gate failed: %s - %s", self.name, e)
            
            return QualityGateResult(
                gate_name=self.name,
                success=False,
                score=0.0,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    async def _run_check(self) -> Tuple[float, Dict[str, Any]]:
        """Override in subclasses to implement specific checks."""
        raise NotImplementedError


class TestSuiteGate(QualityGate):
    """Quality gate for test suite execution."""
    
    def __init__(self):
        super().__init__("test_suite", required_score=0.85)
    
    async def _run_check(self) -> Tuple[float, Dict[str, Any]]:
        """Run the complete test suite."""
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--cov=src/crewai_email_triage",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--cov-fail-under=85",
                "-v",
                "--tb=short"
            ], capture_output=True, text=True, timeout=300)
            
            # Parse coverage report
            coverage_data = {}
            try:
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
            except FileNotFoundError:
                pass
            
            # Calculate score based on test results and coverage
            coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0)
            test_success = result.returncode == 0
            
            # Score: 50% test success, 50% coverage
            score = (0.5 * (1.0 if test_success else 0.0)) + (0.5 * (coverage_percent / 100))
            
            return score, {
                "test_success": test_success,
                "coverage_percent": coverage_percent,
                "stdout": result.stdout[-1000:],  # Last 1000 chars
                "stderr": result.stderr[-1000:],
                "exit_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return 0.0, {"error": "Test suite timed out"}
        except Exception as e:
            return 0.0, {"error": str(e)}


class SecurityScanGate(QualityGate):
    """Quality gate for security scanning."""
    
    def __init__(self):
        super().__init__("security_scan", required_score=0.9)
    
    async def _run_check(self) -> Tuple[float, Dict[str, Any]]:
        """Run security scans."""
        try:
            # Run bandit security scan
            bandit_result = subprocess.run([
                sys.executable, "-m", "bandit",
                "-r", "src/",
                "-f", "json",
                "-o", "bandit_report.json"
            ], capture_output=True, text=True)
            
            # Parse bandit results
            security_issues = []
            try:
                with open("bandit_report.json", "r") as f:
                    bandit_data = json.load(f)
                    security_issues = bandit_data.get("results", [])
            except FileNotFoundError:
                pass
            
            # Run safety check for vulnerabilities
            safety_result = subprocess.run([
                sys.executable, "-m", "safety", "check", "--json"
            ], capture_output=True, text=True)
            
            vulnerabilities = []
            if safety_result.returncode == 0:
                try:
                    safety_data = json.loads(safety_result.stdout)
                    vulnerabilities = safety_data if isinstance(safety_data, list) else []
                except json.JSONDecodeError:
                    pass
            
            # Calculate security score
            high_issues = len([i for i in security_issues if i.get("issue_severity") == "HIGH"])
            medium_issues = len([i for i in security_issues if i.get("issue_severity") == "MEDIUM"])
            critical_vulns = len([v for v in vulnerabilities if v.get("vulnerability") and "critical" in v.get("vulnerability", "").lower()])
            
            # Scoring: penalize high-severity issues heavily
            penalty = (high_issues * 0.2) + (medium_issues * 0.1) + (critical_vulns * 0.3)
            score = max(0.0, 1.0 - penalty)
            
            return score, {
                "security_issues": len(security_issues),
                "high_severity_issues": high_issues,
                "medium_severity_issues": medium_issues,
                "vulnerabilities": len(vulnerabilities),
                "critical_vulnerabilities": critical_vulns,
                "bandit_exit_code": bandit_result.returncode,
                "safety_exit_code": safety_result.returncode
            }
            
        except Exception as e:
            return 0.0, {"error": str(e)}


class PerformanceBenchmarkGate(QualityGate):
    """Quality gate for performance benchmarks."""
    
    def __init__(self):
        super().__init__("performance_benchmark", required_score=0.8)
    
    async def _run_check(self) -> Tuple[float, Dict[str, Any]]:
        """Run performance benchmarks."""
        try:
            from src.crewai_email_triage.scalability import benchmark_performance
            
            # Run performance benchmark
            benchmark_results = benchmark_performance()
            
            # Extract key metrics
            summary = benchmark_results.get("summary", {})
            best_throughput = summary.get("best_throughput", 0)
            
            # Performance thresholds
            excellent_throughput = 100  # items/sec
            good_throughput = 50       # items/sec
            
            # Calculate score based on throughput
            if best_throughput >= excellent_throughput:
                throughput_score = 1.0
            elif best_throughput >= good_throughput:
                throughput_score = 0.7 + (0.3 * (best_throughput - good_throughput) / (excellent_throughput - good_throughput))
            else:
                throughput_score = 0.5 * (best_throughput / good_throughput)
            
            return throughput_score, {
                "best_throughput": best_throughput,
                "best_configuration": summary.get("best_configuration"),
                "test_message_count": summary.get("test_message_count"),
                "benchmark_results": benchmark_results
            }
            
        except Exception as e:
            return 0.0, {"error": str(e)}


class CodeQualityGate(QualityGate):
    """Quality gate for code quality checks."""
    
    def __init__(self):
        super().__init__("code_quality", required_score=0.85)
    
    async def _run_check(self) -> Tuple[float, Dict[str, Any]]:
        """Run code quality checks."""
        try:
            # Run ruff linter
            ruff_result = subprocess.run([
                sys.executable, "-m", "ruff", "check", "src/", "--format=json"
            ], capture_output=True, text=True)
            
            # Parse ruff results
            lint_issues = []
            try:
                lint_issues = json.loads(ruff_result.stdout)
            except json.JSONDecodeError:
                pass
            
            # Run mypy type checking
            mypy_result = subprocess.run([
                sys.executable, "-m", "mypy", "src/", "--json-report", "mypy_report"
            ], capture_output=True, text=True)
            
            # Parse mypy results
            type_errors = 0
            try:
                with open("mypy_report/index.txt", "r") as f:
                    mypy_output = f.read()
                    type_errors = mypy_output.count("error:")
            except FileNotFoundError:
                # Count from stderr if report file not available
                type_errors = ruff_result.stderr.count("error:")
            
            # Calculate quality score
            total_issues = len(lint_issues) + type_errors
            
            # Scoring: penalize each issue
            if total_issues == 0:
                score = 1.0
            elif total_issues <= 5:
                score = 0.9 - (total_issues * 0.02)
            elif total_issues <= 20:
                score = 0.8 - ((total_issues - 5) * 0.01)
            else:
                score = max(0.5, 0.65 - ((total_issues - 20) * 0.005))
            
            return score, {
                "lint_issues": len(lint_issues),
                "type_errors": type_errors,
                "total_issues": total_issues,
                "ruff_exit_code": ruff_result.returncode,
                "mypy_exit_code": mypy_result.returncode
            }
            
        except Exception as e:
            return 0.0, {"error": str(e)}


class SystemIntegrationGate(QualityGate):
    """Quality gate for system integration tests."""
    
    def __init__(self):
        super().__init__("system_integration", required_score=0.9)
    
    async def _run_check(self) -> Tuple[float, Dict[str, Any]]:
        """Run system integration tests."""
        try:
            # Test core functionality
            from src.crewai_email_triage import triage_email, triage_batch
            
            test_messages = [
                "Urgent: Server down, need immediate attention!",
                "Meeting tomorrow at 9 AM",
                "Please review the quarterly report"
            ]
            
            integration_results = []
            
            # Test single message processing
            try:
                result = triage_email(test_messages[0])
                integration_results.append({
                    "test": "single_message",
                    "success": isinstance(result, dict) and "category" in result,
                    "result": result
                })
            except Exception as e:
                integration_results.append({
                    "test": "single_message",
                    "success": False,
                    "error": str(e)
                })
            
            # Test batch processing
            try:
                batch_result = triage_batch(test_messages)
                integration_results.append({
                    "test": "batch_processing",
                    "success": isinstance(batch_result, dict) and "results" in batch_result,
                    "result_count": len(batch_result.get("results", []))
                })
            except Exception as e:
                integration_results.append({
                    "test": "batch_processing",
                    "success": False,
                    "error": str(e)
                })
            
            # Test autonomous evolution
            try:
                evolution_result = run_autonomous_evolution_sync()
                integration_results.append({
                    "test": "autonomous_evolution",
                    "success": evolution_result.success_rate > 0.5,
                    "success_rate": evolution_result.success_rate
                })
            except Exception as e:
                integration_results.append({
                    "test": "autonomous_evolution",
                    "success": False,
                    "error": str(e)
                })
            
            # Calculate integration score
            successful_tests = len([r for r in integration_results if r["success"]])
            total_tests = len(integration_results)
            score = successful_tests / total_tests if total_tests > 0 else 0.0
            
            return score, {
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "integration_results": integration_results
            }
            
        except Exception as e:
            return 0.0, {"error": str(e)}


class ResearchValidationGate(QualityGate):
    """Quality gate for research validation."""
    
    def __init__(self):
        super().__init__("research_validation", required_score=0.7)
    
    async def _run_check(self) -> Tuple[float, Dict[str, Any]]:
        """Run research validation."""
        try:
            # Run evolutionary research
            research_result = run_evolutionary_research_sync()
            
            # Extract key metrics
            summary = research_result.get("research_summary", {})
            success_rate = summary.get("success_rate", 0)
            avg_improvement = summary.get("avg_improvement_rate", 0)
            
            # Research validation criteria
            baseline_score = 0.5  # Base score for any research completion
            success_bonus = success_rate * 0.3  # Up to 30% for successful experiments
            improvement_bonus = min(0.2, avg_improvement * 2)  # Up to 20% for improvements
            
            score = baseline_score + success_bonus + improvement_bonus
            
            return score, {
                "research_success_rate": success_rate,
                "average_improvement": avg_improvement,
                "total_experiments": summary.get("total_experiments", 0),
                "successful_experiments": summary.get("successful_experiments", 0),
                "best_algorithm": research_result.get("best_performing_algorithm")
            }
            
        except Exception as e:
            return 0.0, {"error": str(e)}


class QualityGateOrchestrator:
    """Orchestrates all quality gates."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.gates = [
            TestSuiteGate(),
            SecurityScanGate(),
            CodeQualityGate(),
            PerformanceBenchmarkGate(),
            SystemIntegrationGate(),
            ResearchValidationGate()
        ]
    
    async def run_all_quality_gates(self) -> QualityReport:
        """Run all quality gates."""
        self.logger.info("🚀 Starting comprehensive quality gate validation")
        start_time = time.time()
        
        gate_results = []
        
        # Run gates in parallel where possible
        concurrent_gates = []
        for gate in self.gates:
            concurrent_gates.append(gate.execute())
        
        # Wait for all gates to complete
        results = await asyncio.gather(*concurrent_gates, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                gate_results.append(QualityGateResult(
                    gate_name=self.gates[i].name,
                    success=False,
                    score=0.0,
                    error_message=str(result)
                ))
            else:
                gate_results.append(result)
        
        # Calculate overall results
        total_execution_time = time.time() - start_time
        overall_success = all(r.success for r in gate_results)
        overall_score = sum(r.score for r in gate_results) / len(gate_results) if gate_results else 0.0
        
        report = QualityReport(
            timestamp=time.time(),
            overall_success=overall_success,
            overall_score=overall_score,
            gate_results=gate_results,
            total_execution_time=total_execution_time
        )
        
        # Log summary
        passed = len([r for r in gate_results if r.success])
        total = len(gate_results)
        
        if overall_success:
            self.logger.info("✅ All quality gates PASSED (%d/%d) - Score: %.2f/1.0 - Time: %.2fs",
                           passed, total, overall_score, total_execution_time)
        else:
            self.logger.error("❌ Quality gates FAILED (%d/%d) - Score: %.2f/1.0 - Time: %.2fs",
                            passed, total, overall_score, total_execution_time)
            
            # Log failed gates
            for result in gate_results:
                if not result.success:
                    self.logger.error("  ❌ %s: %.2f/1.0 %s", 
                                    result.gate_name, result.score,
                                    f"({result.error_message})" if result.error_message else "")
        
        return report


async def run_quality_gates() -> QualityReport:
    """Run all quality gates and return report."""
    orchestrator = QualityGateOrchestrator()
    return await orchestrator.run_all_quality_gates()


def run_quality_gates_sync() -> QualityReport:
    """Run quality gates synchronously."""
    return asyncio.run(run_quality_gates())


if __name__ == "__main__":
    # Setup logging
    setup_structured_logging(level=logging.INFO)
    
    # Run quality gates
    report = run_quality_gates_sync()
    
    # Save report
    report_path = Path("quality_gates_report.json")
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    
    print(f"\n📊 Quality Gates Report saved to: {report_path}")
    print(f"Overall Success: {'✅ PASSED' if report.overall_success else '❌ FAILED'}")
    print(f"Overall Score: {report.overall_score:.2f}/1.0")
    print(f"Execution Time: {report.total_execution_time:.2f}s")
    
    # Exit with appropriate code
    sys.exit(0 if report.overall_success else 1)
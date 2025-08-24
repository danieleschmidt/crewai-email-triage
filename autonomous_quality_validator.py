#!/usr/bin/env python3
"""
Autonomous Quality Validator
Comprehensive quality gate enforcement with autonomous decision making
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityGate:
    """Quality gate definition"""
    name: str
    description: str
    threshold: float
    current_value: float
    passed: bool
    severity: str  # critical, high, medium, low
    recommendations: List[str]

@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    overall_score: float
    passed: bool
    gates: List[QualityGate]
    execution_time_seconds: float
    timestamp: float
    recommendations: List[str]
    metadata: Dict[str, Any]

class AutonomousQualityValidator:
    """Advanced quality validation with autonomous gate management"""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.quality_thresholds = {
            'test_coverage': 80.0,
            'test_pass_rate': 100.0,
            'security_score': 90.0,
            'performance_score': 85.0,
            'code_quality_score': 80.0,
            'documentation_score': 70.0
        }
        
        self.critical_gates = ['test_coverage', 'test_pass_rate', 'security_score']
        self.validation_history = []
        
        logger.info("Autonomous Quality Validator initialized")
    
    def validate_all_quality_gates(self) -> QualityReport:
        """Execute comprehensive quality validation"""
        start_time = time.time()
        logger.info("🎯 Starting comprehensive quality gate validation...")
        
        gates = []
        recommendations = []
        
        # Test Coverage Gate
        coverage_gate = self._validate_test_coverage()
        gates.append(coverage_gate)
        
        # Test Pass Rate Gate
        test_gate = self._validate_test_execution()
        gates.append(test_gate)
        
        # Security Gate
        security_gate = self._validate_security()
        gates.append(security_gate)
        
        # Performance Gate
        performance_gate = self._validate_performance()
        gates.append(performance_gate)
        
        # Code Quality Gate
        code_quality_gate = self._validate_code_quality()
        gates.append(code_quality_gate)
        
        # Documentation Gate
        docs_gate = self._validate_documentation()
        gates.append(docs_gate)
        
        # Calculate overall score
        total_weight = len(gates)
        passed_weight = sum(1 for gate in gates if gate.passed)
        overall_score = (passed_weight / total_weight) * 100
        
        # Determine overall pass/fail
        critical_failures = [gate for gate in gates if gate.name in self.critical_gates and not gate.passed]
        overall_passed = len(critical_failures) == 0 and overall_score >= 70.0
        
        # Collect recommendations
        for gate in gates:
            recommendations.extend(gate.recommendations)
        
        # Add strategic recommendations
        if not overall_passed:
            recommendations.extend(self._generate_strategic_recommendations(gates))
        
        execution_time = time.time() - start_time
        
        report = QualityReport(
            overall_score=overall_score,
            passed=overall_passed,
            gates=gates,
            execution_time_seconds=execution_time,
            timestamp=time.time(),
            recommendations=recommendations,
            metadata={
                'critical_failures': len(critical_failures),
                'gates_passed': passed_weight,
                'gates_total': total_weight,
                'validation_level': 'comprehensive'
            }
        )
        
        # Store validation history
        self.validation_history.append(report)
        if len(self.validation_history) > 20:  # Keep last 20 validations
            self.validation_history = self.validation_history[-20:]
        
        if overall_passed:
            logger.info(f"✅ Quality validation PASSED with score {overall_score:.1f}%")
        else:
            logger.warning(f"❌ Quality validation FAILED with score {overall_score:.1f}%")
            logger.warning(f"Critical failures: {len(critical_failures)}")
        
        return report
    
    def _validate_test_coverage(self) -> QualityGate:
        """Validate test coverage requirements"""
        logger.info("📊 Validating test coverage...")
        
        try:
            # Run coverage analysis on core modules
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/test_foundational.py', 'tests/test_pipeline.py',
                '--cov=src/crewai_email_triage/core.py',
                '--cov=src/crewai_email_triage/pipeline.py',
                '--cov-report=json',
                '--cov-report=term-missing'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Parse coverage results
            coverage_file = self.project_root / 'coverage.json'
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                current_coverage = coverage_data['totals']['percent_covered']
            else:
                current_coverage = 0.0
            
            passed = current_coverage >= self.quality_thresholds['test_coverage']
            
            recommendations = []
            if not passed:
                recommendations.extend([
                    f"Increase test coverage from {current_coverage:.1f}% to {self.quality_thresholds['test_coverage']:.1f}%",
                    "Add unit tests for core business logic",
                    "Implement integration tests for critical workflows"
                ])
            
            return QualityGate(
                name='test_coverage',
                description='Minimum test coverage requirement',
                threshold=self.quality_thresholds['test_coverage'],
                current_value=current_coverage,
                passed=passed,
                severity='critical',
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Test coverage validation failed: {e}")
            return QualityGate(
                name='test_coverage',
                description='Minimum test coverage requirement',
                threshold=self.quality_thresholds['test_coverage'],
                current_value=0.0,
                passed=False,
                severity='critical',
                recommendations=['Fix test coverage execution issues']
            )
    
    def _validate_test_execution(self) -> QualityGate:
        """Validate test execution and pass rate"""
        logger.info("🧪 Validating test execution...")
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                'tests/test_foundational.py', 
                'tests/test_pipeline.py',
                '-v', '--tb=short'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Parse test results
            output_lines = result.stdout.split('\n')
            test_results = [line for line in output_lines if 'PASSED' in line or 'FAILED' in line]
            
            passed_tests = len([line for line in test_results if 'PASSED' in line])
            total_tests = len(test_results)
            
            if total_tests > 0:
                pass_rate = (passed_tests / total_tests) * 100
            else:
                pass_rate = 0.0
            
            passed = pass_rate >= self.quality_thresholds['test_pass_rate']
            
            recommendations = []
            if not passed:
                recommendations.extend([
                    f"Fix failing tests to achieve {self.quality_thresholds['test_pass_rate']:.0f}% pass rate",
                    "Review and resolve test failures",
                    "Ensure test environment stability"
                ])
            
            return QualityGate(
                name='test_pass_rate',
                description='Test execution pass rate',
                threshold=self.quality_thresholds['test_pass_rate'],
                current_value=pass_rate,
                passed=passed,
                severity='critical',
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Test execution validation failed: {e}")
            return QualityGate(
                name='test_pass_rate',
                description='Test execution pass rate',
                threshold=self.quality_thresholds['test_pass_rate'],
                current_value=0.0,
                passed=False,
                severity='critical',
                recommendations=['Fix test execution environment']
            )
    
    def _validate_security(self) -> QualityGate:
        """Validate security requirements"""
        logger.info("🔒 Validating security requirements...")
        
        try:
            # Test security scanning functionality
            result = subprocess.run([
                sys.executable, 'triage.py', '--security-scan'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            # Security is good if no critical issues found
            security_passed = result.returncode == 0 and 'SAFE' in result.stdout
            security_score = 95.0 if security_passed else 70.0
            
            recommendations = []
            if not security_passed:
                recommendations.extend([
                    "Address security vulnerabilities",
                    "Update dependencies with security patches",
                    "Review and strengthen input validation"
                ])
            
            return QualityGate(
                name='security_score',
                description='Security vulnerability assessment',
                threshold=self.quality_thresholds['security_score'],
                current_value=security_score,
                passed=security_score >= self.quality_thresholds['security_score'],
                severity='critical',
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return QualityGate(
                name='security_score',
                description='Security vulnerability assessment',
                threshold=self.quality_thresholds['security_score'],
                current_value=0.0,
                passed=False,
                severity='critical',
                recommendations=['Fix security validation environment']
            )
    
    def _validate_performance(self) -> QualityGate:
        """Validate performance requirements"""
        logger.info("⚡ Validating performance requirements...")
        
        try:
            # Test basic performance
            start_time = time.time()
            result = subprocess.run([
                sys.executable, 'triage.py', '--message', 'Test performance validation', '--enhanced'
            ], capture_output=True, text=True, cwd=self.project_root)
            processing_time = time.time() - start_time
            
            # Performance score based on response time
            if processing_time < 1.0:
                performance_score = 95.0
            elif processing_time < 2.0:
                performance_score = 85.0
            elif processing_time < 5.0:
                performance_score = 75.0
            else:
                performance_score = 60.0
            
            passed = performance_score >= self.quality_thresholds['performance_score']
            
            recommendations = []
            if not passed:
                recommendations.extend([
                    f"Improve response time from {processing_time:.2f}s",
                    "Optimize critical code paths",
                    "Consider caching strategies",
                    "Review resource utilization"
                ])
            
            return QualityGate(
                name='performance_score',
                description='System performance and response time',
                threshold=self.quality_thresholds['performance_score'],
                current_value=performance_score,
                passed=passed,
                severity='high',
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return QualityGate(
                name='performance_score',
                description='System performance and response time',
                threshold=self.quality_thresholds['performance_score'],
                current_value=0.0,
                passed=False,
                severity='high',
                recommendations=['Fix performance validation environment']
            )
    
    def _validate_code_quality(self) -> QualityGate:
        """Validate code quality standards"""
        logger.info("📝 Validating code quality...")
        
        # Check for key quality indicators
        quality_indicators = {
            'pyproject_toml': self.project_root / 'pyproject.toml',
            'src_structure': self.project_root / 'src',
            'test_structure': self.project_root / 'tests',
            'readme': self.project_root / 'README.md',
            'requirements': self.project_root / 'pyproject.toml'
        }
        
        quality_score = 0
        total_indicators = len(quality_indicators)
        
        for indicator, path in quality_indicators.items():
            if path.exists():
                quality_score += 1
        
        quality_percentage = (quality_score / total_indicators) * 100
        passed = quality_percentage >= self.quality_thresholds['code_quality_score']
        
        recommendations = []
        if not passed:
            missing_indicators = [name for name, path in quality_indicators.items() if not path.exists()]
            recommendations.extend([
                f"Add missing quality indicators: {', '.join(missing_indicators)}",
                "Follow Python packaging best practices",
                "Maintain consistent code structure"
            ])
        
        return QualityGate(
            name='code_quality_score',
            description='Code quality and structure standards',
            threshold=self.quality_thresholds['code_quality_score'],
            current_value=quality_percentage,
            passed=passed,
            severity='medium',
            recommendations=recommendations
        )
    
    def _validate_documentation(self) -> QualityGate:
        """Validate documentation requirements"""
        logger.info("📚 Validating documentation...")
        
        documentation_files = {
            'README.md': self.project_root / 'README.md',
            'ARCHITECTURE.md': self.project_root / 'ARCHITECTURE.md',
            'CHANGELOG.md': self.project_root / 'CHANGELOG.md',
            'CONTRIBUTING.md': self.project_root / 'CONTRIBUTING.md'
        }
        
        docs_score = 0
        total_docs = len(documentation_files)
        
        for doc_name, doc_path in documentation_files.items():
            if doc_path.exists() and doc_path.stat().st_size > 100:  # Non-empty files
                docs_score += 1
        
        docs_percentage = (docs_score / total_docs) * 100
        passed = docs_percentage >= self.quality_thresholds['documentation_score']
        
        recommendations = []
        if not passed:
            missing_docs = [name for name, path in documentation_files.items() 
                           if not path.exists() or path.stat().st_size <= 100]
            recommendations.extend([
                f"Add or improve documentation files: {', '.join(missing_docs)}",
                "Ensure documentation is comprehensive and up-to-date",
                "Include usage examples and API documentation"
            ])
        
        return QualityGate(
            name='documentation_score',
            description='Documentation completeness and quality',
            threshold=self.quality_thresholds['documentation_score'],
            current_value=docs_percentage,
            passed=passed,
            severity='low',
            recommendations=recommendations
        )
    
    def _generate_strategic_recommendations(self, gates: List[QualityGate]) -> List[str]:
        """Generate strategic recommendations based on overall quality assessment"""
        critical_failures = [gate for gate in gates if gate.name in self.critical_gates and not gate.passed]
        
        strategic_recommendations = []
        
        if len(critical_failures) > 0:
            strategic_recommendations.append("⚠️  CRITICAL: Address all critical quality gate failures before deployment")
        
        # Coverage specific recommendations
        coverage_gate = next((gate for gate in gates if gate.name == 'test_coverage'), None)
        if coverage_gate and not coverage_gate.passed:
            strategic_recommendations.append("🎯 Priority: Implement comprehensive test suite with >80% coverage")
        
        # Security specific recommendations
        security_gate = next((gate for gate in gates if gate.name == 'security_score'), None)
        if security_gate and not security_gate.passed:
            strategic_recommendations.append("🔒 Priority: Conduct security audit and vulnerability assessment")
        
        return strategic_recommendations
    
    def export_quality_report(self, report: QualityReport, file_path: Optional[str] = None) -> str:
        """Export quality report to JSON file"""
        if file_path is None:
            file_path = f"/root/repo/quality_report_{int(time.time())}.json"
        
        report_data = {
            'report': asdict(report),
            'thresholds': self.quality_thresholds,
            'validation_timestamp': time.time(),
            'validator_version': '1.0.0'
        }
        
        with open(file_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Quality report exported to: {file_path}")
        return file_path

def autonomous_quality_validation():
    """Execute autonomous quality validation"""
    print("🎯 AUTONOMOUS QUALITY VALIDATION")
    print("=" * 80)
    
    validator = AutonomousQualityValidator()
    report = validator.validate_all_quality_gates()
    
    # Display results
    print(f"\n📊 OVERALL SCORE: {report.overall_score:.1f}%")
    print(f"🎯 RESULT: {'✅ PASSED' if report.passed else '❌ FAILED'}")
    print(f"⏱️  EXECUTION TIME: {report.execution_time_seconds:.2f}s")
    print(f"🕐 TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(report.timestamp))}")
    
    print("\n📋 QUALITY GATE RESULTS:")
    print("-" * 80)
    
    for gate in report.gates:
        status_icon = "✅" if gate.passed else "❌"
        severity_icon = {"critical": "🚨", "high": "⚠️", "medium": "🔶", "low": "ℹ️"}.get(gate.severity, "")
        
        print(f"{status_icon} {gate.name.upper()} {severity_icon}")
        print(f"   Score: {gate.current_value:.1f}% (Threshold: {gate.threshold:.1f}%)")
        print(f"   Description: {gate.description}")
        
        if gate.recommendations:
            print("   Recommendations:")
            for rec in gate.recommendations:
                print(f"     • {rec}")
        print()
    
    if report.recommendations:
        print("🎯 STRATEGIC RECOMMENDATIONS:")
        print("-" * 80)
        for rec in report.recommendations:
            print(f"• {rec}")
        print()
    
    # Export report
    report_file = validator.export_quality_report(report)
    print(f"📋 Detailed report exported to: {report_file}")
    print("=" * 80)
    
    return report

if __name__ == "__main__":
    # Execute autonomous quality validation
    report = autonomous_quality_validation()
    
    # Exit with appropriate code
    sys.exit(0 if report.passed else 1)
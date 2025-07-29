#!/usr/bin/env python3
"""Test Coverage Optimization Suite.

This script provides comprehensive test coverage analysis and optimization with:
- Detailed coverage analysis and gap identification
- Mutation testing for test quality assessment
- Automated test generation suggestions
- Coverage improvement tracking
"""

import os
import sys
import json
import subprocess
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CoverageGap:
    """Coverage gap data structure."""
    file_path: str
    function_name: str
    line_numbers: List[int]
    gap_type: str  # untested_function, partial_branch, exception_handling
    complexity_score: int
    suggested_tests: List[str]


@dataclass
class MutationTestResult:
    """Mutation testing result data structure."""
    file_path: str
    mutant_id: str
    mutation_type: str
    line_number: int
    original_code: str
    mutated_code: str
    killed: bool
    test_that_killed: Optional[str]


@dataclass
class CoverageReport:
    """Comprehensive coverage report."""
    overall_coverage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    files_analyzed: int
    coverage_gaps: List[CoverageGap]
    improvement_suggestions: List[str]
    target_coverage: float
    estimated_tests_needed: int


class TestCoverageOptimizer:
    """Comprehensive test coverage optimization system."""
    
    def __init__(self, 
                 source_dir: str = "src",
                 test_dir: str = "tests",
                 target_coverage: float = 0.90):
        """Initialize the coverage optimizer.
        
        Args:
            source_dir: Source code directory
            test_dir: Test directory
            target_coverage: Target coverage percentage (0.90 = 90%)
        """
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.target_coverage = target_coverage
        
        # Analysis results
        self.coverage_data = {}
        self.coverage_gaps = []
        self.mutation_results = []
        
        # File patterns to analyze
        self.python_files = list(self.source_dir.glob("**/*.py"))
        self.test_files = list(self.test_dir.glob("**/test_*.py"))
        
        logger.info(f"Found {len(self.python_files)} source files and {len(self.test_files)} test files")
    
    def analyze_coverage(self) -> CoverageReport:
        """Perform comprehensive coverage analysis."""
        logger.info("Starting comprehensive coverage analysis...")
        
        # Run coverage analysis
        coverage_data = self._run_coverage_analysis()
        
        # Identify coverage gaps
        self.coverage_gaps = self._identify_coverage_gaps(coverage_data)
        
        # Calculate metrics
        overall_coverage = coverage_data.get('overall_coverage', 0.0)
        line_coverage = coverage_data.get('line_coverage', 0.0)
        branch_coverage = coverage_data.get('branch_coverage', 0.0)
        function_coverage = coverage_data.get('function_coverage', 0.0)
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions()
        
        # Estimate tests needed
        tests_needed = self._estimate_tests_needed(overall_coverage)
        
        report = CoverageReport(
            overall_coverage=overall_coverage,
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            files_analyzed=len(self.python_files),
            coverage_gaps=self.coverage_gaps,
            improvement_suggestions=suggestions,
            target_coverage=self.target_coverage,
            estimated_tests_needed=tests_needed
        )
        
        logger.info(f"Coverage analysis complete: {overall_coverage:.1%} current, {self.target_coverage:.1%} target")
        return report
    
    def _run_coverage_analysis(self) -> Dict:
        """Run pytest with coverage analysis."""
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=" + str(self.source_dir),
                "--cov-report=xml:coverage.xml",
                "--cov-report=json:coverage.json",
                "--cov-report=html:htmlcov",
                "--cov-branch",
                str(self.test_dir),
                "-v"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                logger.warning(f"Coverage analysis had issues: {result.stderr}")
            
            # Parse coverage results
            return self._parse_coverage_results()
            
        except subprocess.TimeoutExpired:
            logger.error("Coverage analysis timed out")
            return {}
        except Exception as e:
            logger.error(f"Failed to run coverage analysis: {e}")
            return {}
    
    def _parse_coverage_results(self) -> Dict:
        """Parse coverage results from XML and JSON reports."""
        coverage_data = {}
        
        try:
            # Parse JSON coverage report
            if Path("coverage.json").exists():
                with open("coverage.json") as f:
                    json_data = json.load(f)
                    
                    totals = json_data.get("totals", {})
                    coverage_data["line_coverage"] = totals.get("percent_covered", 0.0) / 100
                    coverage_data["lines_covered"] = totals.get("covered_lines", 0)  
                    coverage_data["lines_total"] = totals.get("num_statements", 0)
            
            # Parse XML coverage report for branch coverage
            if Path("coverage.xml").exists():
                tree = ET.parse("coverage.xml")
                root = tree.getroot()
                
                overall_coverage = float(root.get("line-rate", 0)) 
                branch_coverage = float(root.get("branch-rate", 0))
                
                coverage_data["overall_coverage"] = overall_coverage
                coverage_data["branch_coverage"] = branch_coverage
                
                # Count functions with coverage
                functions_covered = 0
                functions_total = 0
                
                for class_elem in root.findall(".//class"):
                    for method_elem in class_elem.findall(".//method"):
                        functions_total += 1
                        if float(method_elem.get("line-rate", 0)) > 0:
                            functions_covered += 1
                
                if functions_total > 0:
                    coverage_data["function_coverage"] = functions_covered / functions_total
                else:
                    coverage_data["function_coverage"] = 1.0
            
            return coverage_data
            
        except Exception as e:
            logger.error(f"Failed to parse coverage results: {e}")
            return {}
    
    def _identify_coverage_gaps(self, coverage_data: Dict) -> List[CoverageGap]:
        """Identify specific coverage gaps that need attention."""
        gaps = []
        
        try:
            # Analyze each source file for coverage gaps
            for python_file in self.python_files:
                file_gaps = self._analyze_file_coverage(python_file)
                gaps.extend(file_gaps)
            
            # Sort by complexity score (highest first)
            gaps.sort(key=lambda x: x.complexity_score, reverse=True)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to identify coverage gaps: {e}")
            return []
    
    def _analyze_file_coverage(self, file_path: Path) -> List[CoverageGap]:
        """Analyze coverage gaps in a specific file."""
        gaps = []
        
        try:
            # Parse the Python file to identify functions and complexity
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Find all function definitions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    gap = self._analyze_function_coverage(file_path, node, content)
                    if gap:
                        gaps.append(gap)
            
            return gaps
            
        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return []
    
    def _analyze_function_coverage(self, file_path: Path, func_node: ast.FunctionDef, content: str) -> Optional[CoverageGap]:
        """Analyze coverage for a specific function."""
        try:
            # Calculate cyclomatic complexity
            complexity = self._calculate_complexity(func_node)
            
            # Check if function is tested
            is_tested = self._is_function_tested(file_path, func_node.name)
            
            if not is_tested or complexity > 3:  # High complexity functions need better testing
                # Generate suggested tests
                suggestions = self._generate_test_suggestions(func_node, complexity)
                
                # Determine gap type
                if not is_tested:
                    gap_type = "untested_function"
                elif complexity > 5:
                    gap_type = "high_complexity"
                else:
                    gap_type = "partial_coverage"
                
                return CoverageGap(
                    file_path=str(file_path),
                    function_name=func_node.name,
                    line_numbers=[func_node.lineno],
                    gap_type=gap_type,
                    complexity_score=complexity,
                    suggested_tests=suggestions
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to analyze function {func_node.name}: {e}")
            return None
    
    def _calculate_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            # Count decision points
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.TryExcept):
                complexity += len(node.handlers)
        
        return complexity
    
    def _is_function_tested(self, file_path: Path, func_name: str) -> bool:
        """Check if a function has corresponding tests."""
        # Simple heuristic: look for test functions that mention the function name
        test_pattern = f"test_{func_name}"
        
        for test_file in self.test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if test_pattern in content or func_name in content:
                        return True
            except Exception:
                continue
        
        return False
    
    def _generate_test_suggestions(self, func_node: ast.FunctionDef, complexity: int) -> List[str]:
        """Generate test suggestions for a function."""
        suggestions = []
        
        # Basic test suggestions
        suggestions.append(f"test_{func_node.name}_happy_path")
        suggestions.append(f"test_{func_node.name}_edge_cases")
        
        # Check for exception handling
        has_exceptions = any(isinstance(node, ast.Raise) for node in ast.walk(func_node))
        if has_exceptions:
            suggestions.append(f"test_{func_node.name}_error_conditions")
        
        # Check for async functions
        if isinstance(func_node, ast.AsyncFunctionDef):
            suggestions.append(f"test_{func_node.name}_async_behavior")
        
        # High complexity functions need more tests
        if complexity > 5:
            suggestions.extend([
                f"test_{func_node.name}_boundary_conditions",
                f"test_{func_node.name}_complex_scenarios",
                f"test_{func_node.name}_performance"
            ])
        
        return suggestions
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """Generate overall coverage improvement suggestions."""
        suggestions = []
        
        # Analyze gap patterns
        gap_types = {}
        high_complexity_functions = []
        
        for gap in self.coverage_gaps:
            gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1
            if gap.complexity_score > 5:
                high_complexity_functions.append(gap)
        
        # Generate specific suggestions
        if gap_types.get("untested_function", 0) > 0:
            suggestions.append(f"Add tests for {gap_types['untested_function']} untested functions")
        
        if gap_types.get("high_complexity", 0) > 0:
            suggestions.append(f"Improve test coverage for {gap_types['high_complexity']} high-complexity functions")
        
        if len(high_complexity_functions) > 0:
            suggestions.append("Consider refactoring high-complexity functions to improve testability")
        
        # File-specific suggestions
        files_with_gaps = set(gap.file_path for gap in self.coverage_gaps)
        if len(files_with_gaps) > 0:
            suggestions.append(f"Focus testing efforts on {len(files_with_gaps)} files with coverage gaps")
        
        # Integration test suggestions
        if len(self.test_files) < len(self.python_files) * 0.5:
            suggestions.append("Consider adding more integration tests")
        
        return suggestions
    
    def _estimate_tests_needed(self, current_coverage: float) -> int:
        """Estimate number of tests needed to reach target coverage."""
        if current_coverage >= self.target_coverage:
            return 0
        
        # Simple heuristic: each test typically improves coverage by 1-3%
        coverage_gap = self.target_coverage - current_coverage
        estimated_improvement_per_test = 0.02  # 2% per test
        
        return max(1, int(coverage_gap / estimated_improvement_per_test))
    
    def run_mutation_testing(self) -> List[MutationTestResult]:
        """Run mutation testing to assess test quality."""
        logger.info("Starting mutation testing analysis...")
        
        try:
            # Use mutmut for mutation testing if available
            result = subprocess.run([
                sys.executable, "-m", "mutmut", "run",
                "--paths-to-mutate", str(self.source_dir),
                "--tests-dir", str(self.test_dir)
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                return self._parse_mutation_results()
            else:
                logger.warning("Mutation testing not available, using simplified analysis")
                return self._simple_mutation_analysis()
                
        except subprocess.TimeoutExpired:
            logger.error("Mutation testing timed out")
            return []
        except FileNotFoundError:
            logger.warning("mutmut not installed, skipping mutation testing")
            return []
        except Exception as e:
            logger.error(f"Mutation testing failed: {e}")
            return []
    
    def _parse_mutation_results(self) -> List[MutationTestResult]:
        """Parse mutation testing results."""
        # This would parse mutmut results if available
        # For now, return empty list as placeholder
        return []
    
    def _simple_mutation_analysis(self) -> List[MutationTestResult]:
        """Perform simplified mutation analysis."""
        # Simplified analysis that looks for common patterns that indicate weak tests
        results = []
        
        for python_file in self.python_files:
            try:
                with open(python_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for patterns that often indicate weak test coverage
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    if any(pattern in line for pattern in ['if ', 'elif ', 'else:', 'try:', 'except:']):
                        # These are potential mutation points
                        result = MutationTestResult(
                            file_path=str(python_file),
                            mutant_id=f"{python_file.name}:{i}",
                            mutation_type="branch_condition",
                            line_number=i,
                            original_code=line.strip(),
                            mutated_code="# mutation would modify this line",
                            killed=False,  # Assume not killed for conservative estimate
                            test_that_killed=None
                        )
                        results.append(result)
            
            except Exception as e:
                logger.error(f"Failed to analyze {python_file}: {e}")
        
        return results
    
    def generate_test_templates(self, coverage_gaps: List[CoverageGap]) -> Dict[str, str]:
        """Generate test template code for coverage gaps."""
        templates = {}
        
        for gap in coverage_gaps[:10]:  # Limit to top 10 gaps
            template = self._create_test_template(gap)
            templates[f"test_{gap.function_name}"] = template
        
        return templates
    
    def _create_test_template(self, gap: CoverageGap) -> str:
        """Create a test template for a specific coverage gap."""
        function_name = gap.function_name
        file_path = gap.file_path
        
        # Extract module path
        module_path = file_path.replace('/', '.').replace('.py', '')
        
        template = f'''"""Test template for {function_name} function."""

import pytest
from unittest.mock import Mock, patch
from {module_path} import {function_name}


class Test{function_name.title()}:
    """Test class for {function_name} function."""
    
    def test_{function_name}_happy_path(self):
        """Test {function_name} with valid inputs."""
        # Arrange
        # TODO: Set up test data
        
        # Act
        # TODO: Call the function
        
        # Assert
        # TODO: Verify expected behavior
        assert True  # Replace with actual assertions
    
    def test_{function_name}_edge_cases(self):
        """Test {function_name} with edge case inputs."""
        # TODO: Test boundary conditions
        pass
    
    def test_{function_name}_error_conditions(self):
        """Test {function_name} error handling."""
        # TODO: Test exception scenarios
        with pytest.raises(Exception):
            pass  # Replace with actual error test
    
    @pytest.mark.parametrize("input_value,expected", [
        # TODO: Add test parameters
        ("test_input", "expected_output"),
    ])
    def test_{function_name}_parametrized(self, input_value, expected):
        """Parametrized test for {function_name}."""
        # TODO: Implement parametrized test
        pass
'''
        
        return template
    
    def create_coverage_report(self, report: CoverageReport) -> str:
        """Create a detailed coverage report."""
        lines = []
        lines.append("# Test Coverage Optimization Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Coverage summary
        lines.append("## Coverage Summary")
        lines.append(f"- **Overall Coverage**: {report.overall_coverage:.1%}")
        lines.append(f"- **Line Coverage**: {report.line_coverage:.1%}")
        lines.append(f"- **Branch Coverage**: {report.branch_coverage:.1%}")
        lines.append(f"- **Function Coverage**: {report.function_coverage:.1%}")
        lines.append(f"- **Target Coverage**: {report.target_coverage:.1%}")
        lines.append(f"- **Files Analyzed**: {report.files_analyzed}")
        lines.append("")
        
        # Coverage gaps
        if report.coverage_gaps:
            lines.append("## Coverage Gaps")
            lines.append("")
            for i, gap in enumerate(report.coverage_gaps[:10], 1):
                lines.append(f"### {i}. {gap.function_name} ({Path(gap.file_path).name})")
                lines.append(f"- **Type**: {gap.gap_type}")
                lines.append(f"- **Complexity Score**: {gap.complexity_score}")
                lines.append(f"- **Line Numbers**: {gap.line_numbers}")
                lines.append("- **Suggested Tests**:")
                for suggestion in gap.suggested_tests:
                    lines.append(f"  - {suggestion}")
                lines.append("")
        
        # Improvement suggestions
        lines.append("## Improvement Suggestions")
        lines.append("")
        for i, suggestion in enumerate(report.improvement_suggestions, 1):
            lines.append(f"{i}. {suggestion}")
        lines.append("")
        
        # Test estimation
        lines.append("## Test Coverage Goals")
        lines.append(f"- **Estimated Tests Needed**: {report.estimated_tests_needed}")
        lines.append(f"- **Current Gap**: {(report.target_coverage - report.overall_coverage):.1%}")
        lines.append("")
        
        return "\n".join(lines)
    
    def export_coverage_metrics(self, report: CoverageReport) -> str:
        """Export coverage metrics in Prometheus format."""
        timestamp = int(datetime.now().timestamp() * 1000)
        
        metrics = []
        metrics.append(f'test_coverage_overall{{target="{report.target_coverage:.2f}"}} {report.overall_coverage:.4f} {timestamp}')
        metrics.append(f'test_coverage_line {{}} {report.line_coverage:.4f} {timestamp}')
        metrics.append(f'test_coverage_branch {{}} {report.branch_coverage:.4f} {timestamp}')
        metrics.append(f'test_coverage_function {{}} {report.function_coverage:.4f} {timestamp}')
        metrics.append(f'test_coverage_gaps_total {{}} {len(report.coverage_gaps)} {timestamp}')
        metrics.append(f'test_coverage_files_analyzed {{}} {report.files_analyzed} {timestamp}')
        metrics.append(f'test_coverage_tests_needed {{}} {report.estimated_tests_needed} {timestamp}')
        
        return '\n'.join(metrics)


def main():
    """Main entry point for test coverage optimizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Coverage Optimizer")
    parser.add_argument("--source-dir", default="src", help="Source code directory")
    parser.add_argument("--test-dir", default="tests", help="Test directory")
    parser.add_argument("--target", type=float, default=0.90, help="Target coverage (0.90 = 90%)")
    parser.add_argument("--mutation-testing", action="store_true", help="Run mutation testing")
    parser.add_argument("--generate-templates", action="store_true", help="Generate test templates")
    parser.add_argument("--export-metrics", action="store_true", help="Export Prometheus metrics")
    parser.add_argument("--output-dir", default="coverage-analysis", help="Output directory for reports")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize optimizer
        optimizer = TestCoverageOptimizer(
            source_dir=args.source_dir,
            test_dir=args.test_dir,
            target_coverage=args.target
        )
        
        # Run coverage analysis
        report = optimizer.analyze_coverage()
        
        # Create coverage report
        report_content = optimizer.create_coverage_report(report)
        report_file = output_dir / f"coverage-report-{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        print(f"Coverage report saved to {report_file}")
        
        # Run mutation testing if requested
        if args.mutation_testing:
            mutation_results = optimizer.run_mutation_testing()
            if mutation_results:
                mutation_file = output_dir / "mutation-test-results.json"
                with open(mutation_file, 'w') as f:
                    json.dump([asdict(result) for result in mutation_results], f, indent=2)
                print(f"Mutation test results saved to {mutation_file}")
        
        # Generate test templates if requested
        if args.generate_templates:
            templates = optimizer.generate_test_templates(report.coverage_gaps)
            templates_dir = output_dir / "test-templates"
            templates_dir.mkdir(exist_ok=True)
            
            for test_name, template_code in templates.items():
                template_file = templates_dir / f"{test_name}.py"
                with open(template_file, 'w') as f:
                    f.write(template_code)
            
            print(f"Test templates generated in {templates_dir}")
        
        # Export metrics if requested
        if args.export_metrics:
            metrics = optimizer.export_coverage_metrics(report)
            metrics_file = output_dir / "coverage-metrics.prom"
            with open(metrics_file, 'w') as f:
                f.write(metrics)
            print(f"Prometheus metrics exported to {metrics_file}")
        
        # Print summary
        print(f"\nCoverage Analysis Summary:")
        print(f"Current Coverage: {report.overall_coverage:.1%}")
        print(f"Target Coverage: {report.target_coverage:.1%}")
        print(f"Coverage Gaps: {len(report.coverage_gaps)}")
        print(f"Estimated Tests Needed: {report.estimated_tests_needed}")
        
        # Exit with appropriate code
        if report.overall_coverage >= report.target_coverage:
            print("✅ Target coverage achieved!")
            return 0
        else:
            print("⚠️ Target coverage not yet achieved")
            return 1
            
    except Exception as e:
        logger.error(f"Coverage optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
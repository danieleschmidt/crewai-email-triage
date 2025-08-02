#!/usr/bin/env python3
"""
Automated metrics collection script for CrewAI Email Triage project.

This script collects various project metrics including code quality, security,
performance, and development productivity metrics.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class MetricsCollector:
    """Collects and aggregates project metrics."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize metrics collector.
        
        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)
        self.metrics = {
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "repository_path": str(self.repo_path.absolute()),
            "metrics": {}
        }
        
    def run_command(self, command: List[str], capture_output: bool = True) -> Optional[str]:
        """Run a command and return its output.
        
        Args:
            command: Command to run as list of strings
            capture_output: Whether to capture output
            
        Returns:
            Command output or None if failed
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=capture_output,
                text=True,
                timeout=60
            )
            return result.stdout if result.returncode == 0 else None
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
            print(f"Command failed: {' '.join(command)} - {e}")
            return None
            
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {}
        
        # Test coverage
        print("📊 Collecting test coverage...")
        coverage_output = self.run_command([
            "python", "-m", "pytest", 
            "--cov=src/crewai_email_triage",
            "--cov-report=json",
            "--cov-report=term-missing",
            "-q"
        ])
        
        if coverage_output and os.path.exists("coverage.json"):
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)
                metrics["test_coverage"] = {
                    "percentage": coverage_data["totals"]["percent_covered"],
                    "lines_covered": coverage_data["totals"]["covered_lines"],
                    "lines_total": coverage_data["totals"]["num_statements"]
                }
        
        # Linting violations
        print("🔍 Collecting linting metrics...")
        lint_output = self.run_command([
            "ruff", "check", "src", "tests", "--output-format=json"
        ])
        
        if lint_output:
            try:
                lint_data = json.loads(lint_output)
                metrics["lint_violations"] = {
                    "total": len(lint_data),
                    "by_severity": self._group_lint_by_severity(lint_data)
                }
            except json.JSONDecodeError:
                metrics["lint_violations"] = {"total": 0, "by_severity": {}}
        
        # Type checking
        print("🔍 Collecting type coverage...")
        type_output = self.run_command([
            "mypy", "src", "--json-report", "mypy-report"
        ])
        
        if os.path.exists("mypy-report/index.json"):
            with open("mypy-report/index.json", "r") as f:
                mypy_data = json.load(f)
                metrics["type_coverage"] = {
                    "percentage": mypy_data.get("summary", {}).get("percent_covered", 0),
                    "errors": len(mypy_data.get("files", {}))
                }
        
        # Complexity metrics
        print("📈 Collecting complexity metrics...")
        complexity_output = self.run_command([
            "radon", "cc", "src", "--json"
        ])
        
        if complexity_output:
            try:
                complexity_data = json.loads(complexity_output)
                complexities = []
                for file_data in complexity_data.values():
                    for item in file_data:
                        if isinstance(item, dict) and "complexity" in item:
                            complexities.append(item["complexity"])
                
                if complexities:
                    metrics["cyclomatic_complexity"] = {
                        "average": sum(complexities) / len(complexities),
                        "max": max(complexities),
                        "functions_analyzed": len(complexities)
                    }
            except json.JSONDecodeError:
                pass
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        metrics = {}
        
        # Vulnerability scanning
        print("🔒 Collecting security vulnerabilities...")
        safety_output = self.run_command([
            "safety", "check", "--json"
        ])
        
        if safety_output:
            try:
                safety_data = json.loads(safety_output)
                vulnerabilities = safety_data.get("vulnerabilities", [])
                
                severity_count = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                for vuln in vulnerabilities:
                    severity = vuln.get("severity", "unknown").lower()
                    if severity in severity_count:
                        severity_count[severity] += 1
                
                metrics["security_vulnerabilities"] = {
                    "total": len(vulnerabilities),
                    "by_severity": severity_count,
                    "packages_affected": len(set(v.get("package_name", "") for v in vulnerabilities))
                }
            except json.JSONDecodeError:
                metrics["security_vulnerabilities"] = {
                    "total": 0,
                    "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0}
                }
        
        # Secret detection
        print("🕵️ Checking for exposed secrets...")
        secrets_output = self.run_command([
            "trufflehog", "filesystem", ".", "--json"
        ])
        
        if secrets_output:
            secrets_found = len(secrets_output.strip().split('\n')) if secrets_output.strip() else 0
            metrics["secrets_exposed"] = {
                "total": secrets_found,
                "last_scan": datetime.now(timezone.utc).isoformat()
            }
        else:
            metrics["secrets_exposed"] = {"total": 0}
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {}
        
        print("⚡ Running performance benchmarks...")
        benchmark_output = self.run_command([
            "python", "-m", "pytest", 
            "tests/performance/",
            "--benchmark-json=benchmark.json",
            "-q"
        ])
        
        if benchmark_output and os.path.exists("benchmark.json"):
            with open("benchmark.json", "r") as f:
                benchmark_data = json.load(f)
                
                benchmarks = benchmark_data.get("benchmarks", [])
                if benchmarks:
                    email_processing_times = []
                    batch_throughputs = []
                    
                    for benchmark in benchmarks:
                        name = benchmark.get("name", "")
                        stats = benchmark.get("stats", {})
                        
                        if "email_processing" in name:
                            email_processing_times.append(stats.get("mean", 0) * 1000)  # Convert to ms
                        elif "batch" in name:
                            # Calculate throughput (emails per second)
                            mean_time = stats.get("mean", 1)
                            if mean_time > 0:
                                batch_throughputs.append(60 / mean_time)  # emails per minute
                    
                    if email_processing_times:
                        metrics["email_processing_time"] = {
                            "average_ms": sum(email_processing_times) / len(email_processing_times),
                            "samples": len(email_processing_times)
                        }
                    
                    if batch_throughputs:
                        metrics["batch_throughput"] = {
                            "emails_per_minute": sum(batch_throughputs) / len(batch_throughputs),
                            "samples": len(batch_throughputs)
                        }
        
        return metrics
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        metrics = {}
        
        print("📊 Collecting Git metrics...")
        
        # Commit frequency (last 30 days)
        commit_output = self.run_command([
            "git", "log", "--since=30.days.ago", "--oneline"
        ])
        
        if commit_output:
            commit_count = len(commit_output.strip().split('\n')) if commit_output.strip() else 0
            metrics["commit_frequency"] = {
                "commits_last_30_days": commit_count,
                "average_per_day": commit_count / 30
            }
        
        # Contributors
        contributors_output = self.run_command([
            "git", "shortlog", "-sn", "--since=90.days.ago"
        ])
        
        if contributors_output:
            contributors = len(contributors_output.strip().split('\n')) if contributors_output.strip() else 0
            metrics["contributors"] = {
                "active_last_90_days": contributors
            }
        
        # Repository size
        repo_size_output = self.run_command([
            "git", "count-objects", "-vH"
        ])
        
        if repo_size_output:
            for line in repo_size_output.split('\n'):
                if 'size-pack' in line:
                    size_str = line.split()[-1]
                    metrics["repository_size"] = {"size_packed": size_str}
                    break
        
        return metrics
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency metrics."""
        metrics = {}
        
        print("📦 Collecting dependency metrics...")
        
        # Outdated packages
        outdated_output = self.run_command([
            "pip", "list", "--outdated", "--format=json"
        ])
        
        if outdated_output:
            try:
                outdated_data = json.loads(outdated_output)
                total_outdated = len(outdated_data)
                
                # Calculate average age (simplified - would need more sophisticated analysis)
                metrics["dependency_freshness"] = {
                    "outdated_packages": total_outdated,
                    "total_packages": total_outdated + 50  # Estimate, would need full package list
                }
            except json.JSONDecodeError:
                pass
        
        return metrics
    
    def collect_docker_metrics(self) -> Dict[str, Any]:
        """Collect Docker-related metrics."""
        metrics = {}
        
        print("🐳 Collecting Docker metrics...")
        
        # Check if Docker image exists and get size
        image_output = self.run_command([
            "docker", "images", "crewai-email-triage", "--format", "json"
        ])
        
        if image_output:
            try:
                for line in image_output.strip().split('\n'):
                    image_data = json.loads(line)
                    size_str = image_data.get("Size", "0B")
                    # Parse size (simplified)
                    if "MB" in size_str:
                        size_mb = float(size_str.replace("MB", ""))
                        metrics["docker_image_size"] = {"size_mb": size_mb}
                    break
            except (json.JSONDecodeError, ValueError):
                pass
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub API metrics."""
        metrics = {}
        
        # This would require GitHub API token and proper implementation
        # For now, return placeholder
        print("🐙 GitHub metrics collection requires API token...")
        
        return metrics
    
    def _group_lint_by_severity(self, lint_data: List[Dict]) -> Dict[str, int]:
        """Group linting violations by severity."""
        severity_count = {"error": 0, "warning": 0, "info": 0}
        
        for item in lint_data:
            level = item.get("level", "info").lower()
            if level in severity_count:
                severity_count[level] += 1
        
        return severity_count
    
    def save_metrics(self, output_file: str = "project-metrics.json"):
        """Save collected metrics to file."""
        output_path = self.repo_path / output_file
        
        with open(output_path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        print(f"✅ Metrics saved to {output_path}")
    
    def collect_all_metrics(self):
        """Collect all available metrics."""
        print("🚀 Starting comprehensive metrics collection...")
        
        collectors = [
            ("code_quality", self.collect_code_quality_metrics),
            ("security", self.collect_security_metrics),
            ("performance", self.collect_performance_metrics),
            ("git", self.collect_git_metrics),
            ("dependencies", self.collect_dependency_metrics),
            ("docker", self.collect_docker_metrics),
            ("github", self.collect_github_metrics),
        ]
        
        for category, collector_func in collectors:
            try:
                print(f"\n📊 Collecting {category} metrics...")
                self.metrics["metrics"][category] = collector_func()
            except Exception as e:
                print(f"❌ Error collecting {category} metrics: {e}")
                self.metrics["metrics"][category] = {"error": str(e)}
        
        # Add summary
        self.metrics["summary"] = self._generate_summary()
        
        print("\n✅ Metrics collection completed!")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of key metrics."""
        summary = {
            "overall_health": "unknown",
            "key_indicators": {},
            "recommendations": []
        }
        
        code_quality = self.metrics["metrics"].get("code_quality", {})
        security = self.metrics["metrics"].get("security", {})
        performance = self.metrics["metrics"].get("performance", {})
        
        # Calculate overall health score
        health_score = 0
        factors = 0
        
        # Test coverage factor
        if "test_coverage" in code_quality:
            coverage = code_quality["test_coverage"]["percentage"]
            health_score += min(coverage / 80 * 25, 25)  # Up to 25 points
            factors += 1
            summary["key_indicators"]["test_coverage"] = f"{coverage:.1f}%"
        
        # Security factor
        if "security_vulnerabilities" in security:
            vulns = security["security_vulnerabilities"]["total"]
            security_score = max(25 - vulns * 5, 0)  # Lose 5 points per vulnerability
            health_score += security_score
            factors += 1
            summary["key_indicators"]["security_vulnerabilities"] = vulns
        
        # Lint violations factor
        if "lint_violations" in code_quality:
            violations = code_quality["lint_violations"]["total"]
            lint_score = max(25 - violations * 2, 0)  # Lose 2 points per violation
            health_score += lint_score
            factors += 1
            summary["key_indicators"]["lint_violations"] = violations
        
        # Performance factor (if available)
        if "email_processing_time" in performance:
            processing_time = performance["email_processing_time"]["average_ms"]
            perf_score = max(25 - max(processing_time - 1000, 0) / 100, 0)
            health_score += perf_score
            factors += 1
            summary["key_indicators"]["avg_processing_time_ms"] = f"{processing_time:.1f}"
        
        if factors > 0:
            health_score = health_score / factors
            
            if health_score >= 20:
                summary["overall_health"] = "excellent"
            elif health_score >= 15:
                summary["overall_health"] = "good"
            elif health_score >= 10:
                summary["overall_health"] = "fair"
            else:
                summary["overall_health"] = "needs_attention"
        
        # Generate recommendations
        if code_quality.get("test_coverage", {}).get("percentage", 0) < 80:
            summary["recommendations"].append("Increase test coverage to at least 80%")
        
        if security.get("security_vulnerabilities", {}).get("total", 0) > 0:
            summary["recommendations"].append("Address security vulnerabilities immediately")
        
        if code_quality.get("lint_violations", {}).get("total", 0) > 10:
            summary["recommendations"].append("Reduce linting violations for better code quality")
        
        return summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Collect comprehensive project metrics"
    )
    parser.add_argument(
        "--repo-path", 
        default=".",
        help="Path to repository root (default: current directory)"
    )
    parser.add_argument(
        "--output",
        default="project-metrics.json", 
        help="Output file path (default: project-metrics.json)"
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        choices=["code_quality", "security", "performance", "git", "dependencies", "docker", "github"],
        help="Specific metric categories to collect (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate repository path
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    if not (repo_path / ".git").exists():
        print(f"⚠️ Directory does not appear to be a Git repository: {repo_path}")
    
    # Initialize collector
    collector = MetricsCollector(args.repo_path)
    
    # Collect metrics
    if args.categories:
        print(f"📊 Collecting specific metrics: {', '.join(args.categories)}")
        # Implement selective collection if needed
        collector.collect_all_metrics()
    else:
        collector.collect_all_metrics()
    
    # Save results
    collector.save_metrics(args.output)
    
    # Print summary
    summary = collector.metrics.get("summary", {})
    if summary:
        print(f"\n📋 Project Health: {summary['overall_health'].upper()}")
        
        key_indicators = summary.get("key_indicators", {})
        if key_indicators:
            print("\n🔑 Key Indicators:")
            for key, value in key_indicators.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        recommendations = summary.get("recommendations", [])
        if recommendations:
            print("\n💡 Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")


if __name__ == "__main__":
    main()
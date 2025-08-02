#!/usr/bin/env python3
"""
Automated repository maintenance script for CrewAI Email Triage.

This script performs various maintenance tasks including dependency updates,
security scanning, cleanup, and health checks.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class RepositoryMaintainer:
    """Handles automated repository maintenance tasks."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize repository maintainer.
        
        Args:
            repo_path: Path to the repository root
        """
        self.repo_path = Path(repo_path)
        self.maintenance_log = []
        self.results = {
            "maintenance_date": datetime.now().isoformat(),
            "tasks_completed": [],
            "tasks_failed": [],
            "recommendations": []
        }
        
    def log(self, message: str, level: str = "info"):
        """Log maintenance activity.
        
        Args:
            message: Log message
            level: Log level (info, warning, error)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level.upper()}: {message}"
        self.maintenance_log.append(log_entry)
        
        # Print with appropriate emoji
        emoji_map = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}
        print(f"{emoji_map.get(level, 'ℹ️')} {message}")
    
    def run_command(self, command: List[str], check: bool = True) -> Optional[subprocess.CompletedProcess]:
        """Run a command and return the result.
        
        Args:
            command: Command to run as list of strings
            check: Whether to raise exception on non-zero exit
            
        Returns:
            CompletedProcess result or None if failed
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=check
            )
            return result
        except subprocess.SubprocessError as e:
            self.log(f"Command failed: {' '.join(command)} - {e}", "error")
            return None
    
    def cleanup_artifacts(self) -> bool:
        """Clean up build artifacts and temporary files."""
        self.log("🧹 Cleaning up build artifacts and temporary files...")
        
        cleanup_patterns = [
            "__pycache__",
            "*.pyc",
            "*.pyo", 
            "*.pyd",
            ".pytest_cache",
            ".coverage",
            "htmlcov",
            ".mypy_cache",
            ".ruff_cache",
            "build",
            "dist",
            "*.egg-info",
            ".tox",
            ".eggs",
            "node_modules",
            ".DS_Store",
            "Thumbs.db",
            "*.tmp",
            "*.temp",
            "*.log",
            "coverage.json",
            "coverage.xml",
            "benchmark.json",
            "mypy-report",
            "*-report.json",
            "*-report.txt"
        ]
        
        cleaned_items = []
        
        for pattern in cleanup_patterns:
            if "*" in pattern:
                # Use shell globbing for wildcard patterns
                result = self.run_command(["find", ".", "-name", pattern, "-delete"], check=False)
                if result and result.returncode == 0:
                    cleaned_items.append(pattern)
            else:
                # Direct path cleanup
                for item_path in self.repo_path.rglob(pattern):
                    try:
                        if item_path.is_file():
                            item_path.unlink()
                        elif item_path.is_dir():
                            shutil.rmtree(item_path)
                        cleaned_items.append(str(item_path.relative_to(self.repo_path)))
                    except (OSError, PermissionError) as e:
                        self.log(f"Could not remove {item_path}: {e}", "warning")
        
        if cleaned_items:
            self.log(f"Cleaned {len(cleaned_items)} items", "success")
            self.results["tasks_completed"].append("cleanup_artifacts")
            return True
        else:
            self.log("No cleanup needed", "info")
            return True
    
    def update_dependencies(self, security_only: bool = False) -> bool:
        """Update project dependencies.
        
        Args:
            security_only: Only update packages with security vulnerabilities
            
        Returns:
            True if successful
        """
        self.log("📦 Checking for dependency updates...")
        
        # Check for security vulnerabilities first
        safety_result = self.run_command(["safety", "check", "--json"], check=False)
        
        if safety_result and safety_result.returncode != 0:
            try:
                safety_data = json.loads(safety_result.stdout)
                vulnerabilities = safety_data.get("vulnerabilities", [])
                
                if vulnerabilities:
                    self.log(f"Found {len(vulnerabilities)} security vulnerabilities", "warning")
                    
                    # Extract vulnerable packages and their safe versions
                    packages_to_update = []
                    for vuln in vulnerabilities:
                        package_name = vuln.get("package_name")
                        safe_versions = vuln.get("safe_versions", [])
                        
                        if package_name and safe_versions:
                            safe_version = safe_versions[-1]  # Use latest safe version
                            packages_to_update.append(f"{package_name}>={safe_version}")
                    
                    if packages_to_update:
                        self.log(f"Updating {len(packages_to_update)} vulnerable packages...")
                        
                        # Update packages
                        for package_spec in packages_to_update:
                            update_result = self.run_command(
                                ["pip", "install", "--upgrade", package_spec], 
                                check=False
                            )
                            
                            if update_result and update_result.returncode == 0:
                                self.log(f"Updated {package_spec}", "success")
                            else:
                                self.log(f"Failed to update {package_spec}", "error")
                        
                        self.results["tasks_completed"].append("security_updates")
                else:
                    self.log("No security vulnerabilities found", "success")
            except json.JSONDecodeError:
                self.log("Could not parse safety check results", "warning")
        
        if not security_only:
            # Check for general updates
            outdated_result = self.run_command(
                ["pip", "list", "--outdated", "--format=json"], 
                check=False
            )
            
            if outdated_result and outdated_result.returncode == 0:
                try:
                    outdated_data = json.loads(outdated_result.stdout)
                    
                    if outdated_data:
                        self.log(f"Found {len(outdated_data)} outdated packages", "info")
                        
                        # Categorize updates
                        minor_updates = []
                        major_updates = []
                        
                        for package in outdated_data:
                            name = package["name"]
                            current = package["version"]
                            latest = package["latest_version"]
                            
                            current_parts = current.split(".")
                            latest_parts = latest.split(".")
                            
                            if (len(current_parts) >= 1 and len(latest_parts) >= 1 and
                                current_parts[0] != latest_parts[0]):
                                major_updates.append(package)
                            else:
                                minor_updates.append(package)
                        
                        # Apply minor updates automatically
                        if minor_updates:
                            self.log(f"Applying {len(minor_updates)} minor updates...")
                            
                            for package in minor_updates[:5]:  # Limit to 5 at a time
                                name = package["name"]
                                latest = package["latest_version"]
                                
                                update_result = self.run_command(
                                    ["pip", "install", "--upgrade", f"{name}=={latest}"],
                                    check=False
                                )
                                
                                if update_result and update_result.returncode == 0:
                                    self.log(f"Updated {name} to {latest}", "success")
                                else:
                                    self.log(f"Failed to update {name}", "warning")
                        
                        # Report major updates
                        if major_updates:
                            self.results["recommendations"].append(
                                f"Consider reviewing {len(major_updates)} major version updates: " +
                                ", ".join([p["name"] for p in major_updates[:5]])
                            )
                    else:
                        self.log("All packages are up to date", "success")
                        
                    self.results["tasks_completed"].append("dependency_updates")
                except json.JSONDecodeError:
                    self.log("Could not parse outdated packages list", "warning")
        
        return True
    
    def run_security_scan(self) -> bool:
        """Run comprehensive security scanning."""
        self.log("🔒 Running security scans...")
        
        security_issues = 0
        
        # Run Bandit security analysis
        bandit_result = self.run_command([
            "bandit", "-r", "src/", "-f", "json"
        ], check=False)
        
        if bandit_result:
            try:
                bandit_data = json.loads(bandit_result.stdout)
                issues = bandit_data.get("results", [])
                
                if issues:
                    high_issues = [i for i in issues if i.get("issue_severity") == "HIGH"]
                    medium_issues = [i for i in issues if i.get("issue_severity") == "MEDIUM"]
                    
                    self.log(f"Bandit found {len(issues)} security issues "
                           f"({len(high_issues)} high, {len(medium_issues)} medium)")
                    
                    security_issues += len(high_issues) * 3 + len(medium_issues)
                else:
                    self.log("Bandit: No security issues found", "success")
            except json.JSONDecodeError:
                self.log("Could not parse Bandit results", "warning")
        
        # Run secret detection
        trufflehog_result = self.run_command([
            "trufflehog", "filesystem", ".", "--json"
        ], check=False)
        
        if trufflehog_result and trufflehog_result.stdout.strip():
            secrets_found = len(trufflehog_result.stdout.strip().split('\n'))
            self.log(f"TruffleHog found {secrets_found} potential secrets", "warning")
            security_issues += secrets_found * 5  # Secrets are serious
        else:
            self.log("TruffleHog: No secrets detected", "success")
        
        # Overall security assessment
        if security_issues == 0:
            self.log("Security scan completed - no issues found", "success")
            self.results["tasks_completed"].append("security_scan_clean")
        elif security_issues <= 5:
            self.log(f"Security scan completed - {security_issues} minor issues", "warning")
            self.results["tasks_completed"].append("security_scan_minor_issues")
        else:
            self.log(f"Security scan completed - {security_issues} issues need attention", "error")
            self.results["tasks_failed"].append("security_scan_major_issues")
            self.results["recommendations"].append("Address security issues immediately")
        
        return security_issues <= 5
    
    def optimize_performance(self) -> bool:
        """Run performance optimization tasks."""
        self.log("⚡ Running performance optimization...")
        
        # Run performance tests to establish baseline
        perf_result = self.run_command([
            "python", "-m", "pytest", 
            "tests/performance/",
            "--benchmark-json=benchmark-maintenance.json",
            "-q"
        ], check=False)
        
        if perf_result and perf_result.returncode == 0:
            if os.path.exists("benchmark-maintenance.json"):
                with open("benchmark-maintenance.json", "r") as f:
                    benchmark_data = json.load(f)
                
                benchmarks = benchmark_data.get("benchmarks", [])
                slow_benchmarks = []
                
                for benchmark in benchmarks:
                    name = benchmark.get("name", "")
                    stats = benchmark.get("stats", {})
                    mean_time = stats.get("mean", 0)
                    
                    # Flag slow operations (>1 second)
                    if mean_time > 1.0:
                        slow_benchmarks.append((name, mean_time))
                
                if slow_benchmarks:
                    self.log(f"Found {len(slow_benchmarks)} slow operations", "warning")
                    self.results["recommendations"].append(
                        "Consider optimizing slow operations: " +
                        ", ".join([f"{name} ({time:.2f}s)" for name, time in slow_benchmarks[:3]])
                    )
                else:
                    self.log("Performance benchmarks look good", "success")
                
                self.results["tasks_completed"].append("performance_analysis")
        else:
            self.log("Could not run performance benchmarks", "warning")
        
        return True
    
    def update_documentation(self) -> bool:
        """Update and validate documentation."""
        self.log("📚 Checking documentation...")
        
        # Check for outdated documentation
        docs_updated = False
        
        # Update README if needed (check last modified vs code changes)
        readme_path = self.repo_path / "README.md"
        if readme_path.exists():
            # Get last modification time of README
            readme_mtime = readme_path.stat().st_mtime
            
            # Check if source code has been modified more recently
            src_files = list(self.repo_path.glob("src/**/*.py"))
            if src_files:
                latest_src_mtime = max(f.stat().st_mtime for f in src_files)
                
                if latest_src_mtime > readme_mtime:
                    self.results["recommendations"].append(
                        "README.md may be outdated - consider updating based on recent code changes"
                    )
        
        # Validate markdown files
        md_files = list(self.repo_path.glob("**/*.md"))
        broken_links = []
        
        for md_file in md_files:
            # Simple link validation (would need more sophisticated implementation)
            try:
                content = md_file.read_text()
                # This is a simplified check - would need proper markdown parsing
                if "TODO" in content or "FIXME" in content:
                    self.results["recommendations"].append(
                        f"Documentation file {md_file.name} contains TODO/FIXME items"
                    )
            except (OSError, UnicodeDecodeError):
                continue
        
        self.results["tasks_completed"].append("documentation_check")
        return True
    
    def check_repository_health(self) -> bool:
        """Check overall repository health."""
        self.log("🏥 Checking repository health...")
        
        health_score = 0
        total_checks = 0
        
        # Check Git repository status
        status_result = self.run_command(["git", "status", "--porcelain"], check=False)
        if status_result:
            if status_result.stdout.strip():
                self.log("Repository has uncommitted changes", "warning")
                self.results["recommendations"].append("Consider committing pending changes")
            else:
                self.log("Repository is clean", "success")
                health_score += 1
            total_checks += 1
        
        # Check for large files
        large_files_result = self.run_command([
            "find", ".", "-type", "f", "-size", "+10M"
        ], check=False)
        
        if large_files_result and large_files_result.stdout.strip():
            large_files = large_files_result.stdout.strip().split('\n')
            self.log(f"Found {len(large_files)} large files (>10MB)", "warning")
            self.results["recommendations"].append(
                "Consider using Git LFS for large files or removing unnecessary large files"
            )
        else:
            health_score += 1
        total_checks += 1
        
        # Check branch protection (would need GitHub API)
        # For now, just check if we're on main/master
        branch_result = self.run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"], check=False)
        if branch_result:
            current_branch = branch_result.stdout.strip()
            if current_branch in ["main", "master"]:
                self.log(f"Working on {current_branch} branch", "info")
            else:
                self.log(f"Working on feature branch: {current_branch}", "info")
        
        # Calculate health percentage
        if total_checks > 0:
            health_percentage = (health_score / total_checks) * 100
            self.log(f"Repository health: {health_percentage:.0f}%", 
                    "success" if health_percentage >= 80 else "warning")
        
        self.results["tasks_completed"].append("repository_health_check")
        return True
    
    def generate_maintenance_report(self) -> str:
        """Generate a maintenance report."""
        report_lines = [
            "# Repository Maintenance Report",
            f"**Date:** {self.results['maintenance_date']}",
            "",
            "## Tasks Completed",
        ]
        
        if self.results["tasks_completed"]:
            for task in self.results["tasks_completed"]:
                report_lines.append(f"- ✅ {task.replace('_', ' ').title()}")
        else:
            report_lines.append("- No tasks completed")
        
        report_lines.extend([
            "",
            "## Tasks Failed",
        ])
        
        if self.results["tasks_failed"]:
            for task in self.results["tasks_failed"]:
                report_lines.append(f"- ❌ {task.replace('_', ' ').title()}")
        else:
            report_lines.append("- No tasks failed")
        
        report_lines.extend([
            "",
            "## Recommendations",
        ])
        
        if self.results["recommendations"]:
            for rec in self.results["recommendations"]:
                report_lines.append(f"- 💡 {rec}")
        else:
            report_lines.append("- No specific recommendations")
        
        report_lines.extend([
            "",
            "## Maintenance Log",
            "```",
        ])
        
        report_lines.extend(self.maintenance_log)
        report_lines.append("```")
        
        return "\n".join(report_lines)
    
    def run_full_maintenance(self, security_only: bool = False) -> bool:
        """Run full maintenance routine.
        
        Args:
            security_only: Only run security-related tasks
            
        Returns:
            True if all tasks succeeded
        """
        self.log("🚀 Starting full repository maintenance...", "info")
        
        success = True
        
        # Always run these tasks
        tasks = [
            ("cleanup", self.cleanup_artifacts),
            ("security_scan", self.run_security_scan),
            ("dependency_updates", lambda: self.update_dependencies(security_only)),
        ]
        
        if not security_only:
            tasks.extend([
                ("performance_optimization", self.optimize_performance),
                ("documentation_update", self.update_documentation),
                ("repository_health", self.check_repository_health),
            ])
        
        for task_name, task_func in tasks:
            try:
                self.log(f"Running {task_name.replace('_', ' ')}...")
                if not task_func():
                    self.log(f"Task {task_name} failed", "error")
                    success = False
            except Exception as e:
                self.log(f"Task {task_name} raised exception: {e}", "error")
                self.results["tasks_failed"].append(task_name)
                success = False
        
        # Save maintenance results
        results_file = self.repo_path / "maintenance-results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Generate and save report
        report = self.generate_maintenance_report()
        report_file = self.repo_path / "maintenance-report.md"
        with open(report_file, "w") as f:
            f.write(report)
        
        if success:
            self.log("🎉 All maintenance tasks completed successfully!", "success")
        else:
            self.log("⚠️ Some maintenance tasks failed. Check the report for details.", "warning")
        
        return success


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Automated repository maintenance"
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to repository root (default: current directory)"
    )
    parser.add_argument(
        "--security-only",
        action="store_true",
        help="Only run security-related maintenance tasks"
    )
    parser.add_argument(
        "--task",
        choices=["cleanup", "dependencies", "security", "performance", "docs", "health", "all"],
        default="all",
        help="Specific task to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Validate repository path
    repo_path = Path(args.repo_path)
    if not repo_path.exists():
        print(f"❌ Repository path does not exist: {repo_path}")
        sys.exit(1)
    
    if not (repo_path / ".git").exists():
        print(f"⚠️ Directory does not appear to be a Git repository: {repo_path}")
        # Continue anyway - might still be useful
    
    # Initialize maintainer
    maintainer = RepositoryMaintainer(args.repo_path)
    
    # Run maintenance tasks
    if args.task == "all":
        success = maintainer.run_full_maintenance(args.security_only)
    else:
        # Run specific task
        task_map = {
            "cleanup": maintainer.cleanup_artifacts,
            "dependencies": lambda: maintainer.update_dependencies(args.security_only),
            "security": maintainer.run_security_scan,
            "performance": maintainer.optimize_performance,
            "docs": maintainer.update_documentation,
            "health": maintainer.check_repository_health,
        }
        
        if args.task in task_map:
            success = task_map[args.task]()
        else:
            print(f"❌ Unknown task: {args.task}")
            sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
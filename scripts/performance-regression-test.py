#!/usr/bin/env python3
"""Advanced Performance Regression Testing Suite.

This script implements comprehensive performance regression testing with:
- Automated benchmarking against baselines
- Performance alerting thresholds
- Regression detection and reporting
- CI/CD integration for continuous monitoring
"""

import json
import time
import statistics
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tests.test_performance_benchmarks import PerformanceBenchmark, run_performance_benchmarks


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    test_name: str
    mean_time: float
    median_time: float
    p95_time: float
    p99_time: float
    std_dev: float
    min_time: float
    max_time: float
    samples: int
    timestamp: str
    git_commit: Optional[str] = None
    baseline_comparison: Optional[float] = None
    regression_detected: bool = False


@dataclass
class PerformanceBaseline:
    """Performance baseline data structure."""
    test_name: str
    baseline_mean: float
    baseline_p95: float
    acceptable_degradation: float
    last_updated: str
    git_commit: str


class PerformanceRegressionDetector:
    """Advanced performance regression detection and alerting."""
    
    def __init__(self, 
                 baselines_file: str = "performance-baselines.json",
                 results_dir: str = "performance-results",
                 regression_threshold: float = 0.15):  # 15% degradation threshold
        """Initialize the regression detector.
        
        Args:
            baselines_file: Path to performance baselines JSON file
            results_dir: Directory to store performance results
            regression_threshold: Default regression threshold (15% = 0.15)
        """
        self.baselines_file = Path(baselines_file)
        self.results_dir = Path(results_dir)
        self.regression_threshold = regression_threshold
        self.results_dir.mkdir(exist_ok=True)
        
        # Load existing baselines
        self.baselines = self._load_baselines()
    
    def _load_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from file."""
        if not self.baselines_file.exists():
            return {}
            
        try:
            with open(self.baselines_file, 'r') as f:
                data = json.load(f)
                return {
                    name: PerformanceBaseline(**baseline)
                    for name, baseline in data.items()
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load baselines: {e}")
            return {}
    
    def _save_baselines(self):
        """Save performance baselines to file."""
        data = {
            name: asdict(baseline)
            for name, baseline in self.baselines.items()
        }
        with open(self.baselines_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None
    
    def run_benchmarks(self) -> List[PerformanceMetrics]:
        """Run performance benchmarks and collect metrics."""
        print("🚀 Running performance benchmarks...")
        
        # Enhanced benchmark configuration
        benchmarks = [
            ("email_classification", self._benchmark_email_classification, 50),
            ("email_prioritization", self._benchmark_email_prioritization, 50),
            ("email_summarization", self._benchmark_email_summarization, 30),
            ("email_response_generation", self._benchmark_email_response, 30),
            ("batch_processing", self._benchmark_batch_processing, 20),
            ("pipeline_end_to_end", self._benchmark_pipeline_e2e, 15),
        ]
        
        metrics = []
        git_commit = self._get_git_commit()
        
        for test_name, benchmark_func, samples in benchmarks:
            print(f"📊 Running {test_name} benchmark ({samples} samples)...")
            
            times = []
            for i in range(samples):
                start_time = time.perf_counter()
                try:
                    benchmark_func()
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                except Exception as e:
                    print(f"❌ Benchmark {test_name} failed on sample {i+1}: {e}")
                    continue
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   Completed {i+1}/{samples} samples")
            
            if not times:
                print(f"⚠️ No successful samples for {test_name}")
                continue
            
            # Calculate statistics
            mean_time = statistics.mean(times)
            median_time = statistics.median(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)
            
            # Calculate percentiles
            sorted_times = sorted(times)
            p95_idx = int(0.95 * len(sorted_times))
            p99_idx = int(0.99 * len(sorted_times))
            p95_time = sorted_times[p95_idx]
            p99_time = sorted_times[p99_idx]
            
            # Check for regression
            baseline_comparison = None
            regression_detected = False
            
            if test_name in self.baselines:
                baseline = self.baselines[test_name]
                baseline_comparison = (mean_time - baseline.baseline_mean) / baseline.baseline_mean
                
                # Check if regression exceeds threshold
                threshold = baseline.acceptable_degradation
                if baseline_comparison > threshold:
                    regression_detected = True
                    print(f"🚨 REGRESSION DETECTED in {test_name}: "
                          f"{baseline_comparison:.1%} degradation "
                          f"(threshold: {threshold:.1%})")
            
            metric = PerformanceMetrics(
                test_name=test_name,
                mean_time=mean_time,
                median_time=median_time,
                p95_time=p95_time,
                p99_time=p99_time,
                std_dev=std_dev,
                min_time=min_time,
                max_time=max_time,
                samples=len(times),
                timestamp=datetime.now().isoformat(),
                git_commit=git_commit,
                baseline_comparison=baseline_comparison,
                regression_detected=regression_detected
            )
            
            metrics.append(metric)
            
            print(f"✅ {test_name}: {mean_time:.3f}s avg, {p95_time:.3f}s p95")
            if baseline_comparison is not None:
                change_indicator = "📈" if baseline_comparison > 0 else "📉"
                print(f"   {change_indicator} {baseline_comparison:+.1%} vs baseline")
        
        return metrics
    
    def _benchmark_email_classification(self):
        """Benchmark email classification performance."""
        from crewai_email_triage import ClassifierAgent
        
        agent = ClassifierAgent()
        test_email = {
            "subject": "Urgent: Server downtime issue needs immediate attention",
            "body": "Our production server has been experiencing intermittent downtime. "
                   "This is affecting customer experience and needs urgent resolution.",
            "sender": "ops@company.com"
        }
        agent.classify(test_email)
    
    def _benchmark_email_prioritization(self):
        """Benchmark email prioritization performance."""
        from crewai_email_triage import PriorityAgent
        
        agent = PriorityAgent()
        test_email = {
            "subject": "Urgent: Server downtime issue needs immediate attention",
            "body": "Our production server has been experiencing intermittent downtime.",
            "sender": "ops@company.com",
            "classification": "technical_issue"
        }
        agent.prioritize(test_email)
    
    def _benchmark_email_summarization(self):
        """Benchmark email summarization performance."""
        from crewai_email_triage import SummarizerAgent
        
        agent = SummarizerAgent()
        test_email = {
            "subject": "Quarterly Business Review Meeting",
            "body": "I hope this email finds you well. I wanted to reach out regarding "
                   "our upcoming quarterly business review meeting scheduled for next week. "
                   "We'll be covering performance metrics, strategic initiatives, and budget planning.",
            "sender": "manager@company.com"
        }
        agent.summarize(test_email)
    
    def _benchmark_email_response(self):
        """Benchmark email response generation performance."""
        from crewai_email_triage import ResponseAgent
        
        agent = ResponseAgent()
        test_email = {
            "subject": "Request for project status update",
            "body": "Could you please provide an update on the current project status?",
            "sender": "stakeholder@company.com",
            "priority": "medium",
            "classification": "request"
        }
        agent.generate_response(test_email)
    
    def _benchmark_batch_processing(self):
        """Benchmark batch email processing performance."""
        from crewai_email_triage.pipeline import triage_batch
        
        test_emails = [
            {
                "subject": f"Test email {i}",
                "body": f"This is test email number {i} for batch processing.",
                "sender": f"test{i}@company.com"
            }
            for i in range(10)  # Process 10 emails in batch
        ]
        triage_batch(test_emails)
    
    def _benchmark_pipeline_e2e(self):
        """Benchmark end-to-end pipeline performance."""
        from crewai_email_triage import triage_email
        
        test_email = {
            "subject": "Customer complaint about service quality",
            "body": "I am writing to express my dissatisfaction with the recent service. "
                   "The response time was unacceptable and the quality was below expectations. "
                   "I expect a prompt resolution to this matter.",
            "sender": "customer@external.com"
        }
        triage_email(test_email)
    
    def save_results(self, metrics: List[PerformanceMetrics], filename: Optional[str] = None):
        """Save performance results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_results_{timestamp}.json"
        
        results_path = self.results_dir / filename
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": self._get_git_commit(),
            "regression_threshold": self.regression_threshold,
            "metrics": [asdict(metric) for metric in metrics]
        }
        
        with open(results_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📄 Results saved to {results_path}")
        return results_path
    
    def update_baselines(self, metrics: List[PerformanceMetrics], force: bool = False):
        """Update performance baselines with new metrics."""
        git_commit = self._get_git_commit()
        updated_count = 0
        
        for metric in metrics:
            # Only update baselines if performance is acceptable
            if metric.regression_detected and not force:
                print(f"⚠️ Skipping baseline update for {metric.test_name} due to regression")
                continue
            
            # Update or create baseline
            baseline = PerformanceBaseline(
                test_name=metric.test_name,
                baseline_mean=metric.mean_time,
                baseline_p95=metric.p95_time,
                acceptable_degradation=self.regression_threshold,
                last_updated=metric.timestamp,
                git_commit=git_commit or "unknown"
            )
            
            self.baselines[metric.test_name] = baseline
            updated_count += 1
            print(f"📊 Updated baseline for {metric.test_name}")
        
        if updated_count > 0:
            self._save_baselines()
            print(f"✅ Updated {updated_count} performance baselines")
        
        return updated_count
    
    def generate_report(self, metrics: List[PerformanceMetrics]) -> str:
        """Generate a performance report."""
        regressions = [m for m in metrics if m.regression_detected]
        
        report = []
        report.append("# Performance Regression Test Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Git Commit: {self._get_git_commit() or 'Unknown'}")
        report.append("")
        
        if regressions:
            report.append("## 🚨 Regressions Detected")
            report.append("")
            for metric in regressions:
                report.append(f"- **{metric.test_name}**: "
                             f"{metric.baseline_comparison:+.1%} degradation "
                             f"({metric.mean_time:.3f}s avg)")
            report.append("")
        else:
            report.append("## ✅ No Regressions Detected")
            report.append("")
        
        report.append("## Performance Summary")
        report.append("")
        report.append("| Test | Mean (s) | P95 (s) | P99 (s) | Std Dev | Baseline Δ |")
        report.append("|------|----------|---------|---------|---------|------------|")
        
        for metric in metrics:
            baseline_change = ""
            if metric.baseline_comparison is not None:
                baseline_change = f"{metric.baseline_comparison:+.1%}"
            
            report.append(
                f"| {metric.test_name} | {metric.mean_time:.3f} | "
                f"{metric.p95_time:.3f} | {metric.p99_time:.3f} | "
                f"{metric.std_dev:.3f} | {baseline_change} |"
            )
        
        return "\n".join(report)
    
    def export_prometheus_metrics(self, metrics: List[PerformanceMetrics], output_file: str = "performance_metrics.prom"):
        """Export metrics in Prometheus format for monitoring integration."""
        prometheus_metrics = []
        timestamp = int(time.time() * 1000)  # Prometheus expects milliseconds
        
        for metric in metrics:
            test_name = metric.test_name.replace('-', '_')
            
            # Mean response time
            prometheus_metrics.append(
                f'performance_test_duration_seconds{{test="{test_name}",percentile="mean"}} '
                f'{metric.mean_time} {timestamp}'
            )
            
            # P95 response time
            prometheus_metrics.append(
                f'performance_test_duration_seconds{{test="{test_name}",percentile="p95"}} '
                f'{metric.p95_time} {timestamp}'
            )
            
            # P99 response time
            prometheus_metrics.append(
                f'performance_test_duration_seconds{{test="{test_name}",percentile="p99"}} '
                f'{metric.p99_time} {timestamp}'
            )
            
            # Regression indicator
            regression_value = 1 if metric.regression_detected else 0
            prometheus_metrics.append(
                f'performance_test_regression_detected{{test="{test_name}"}} '
                f'{regression_value} {timestamp}'
            )
            
            # Baseline comparison (if available)
            if metric.baseline_comparison is not None:
                prometheus_metrics.append(
                    f'performance_test_baseline_change_ratio{{test="{test_name}"}} '
                    f'{metric.baseline_comparison} {timestamp}'
                )
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(prometheus_metrics))
        
        print(f"📊 Prometheus metrics exported to {output_file}")
        return output_file


def main():
    """Main entry point for performance regression testing."""
    parser = argparse.ArgumentParser(description="Performance Regression Testing Suite")
    parser.add_argument("--update-baselines", action="store_true",
                       help="Update performance baselines with current results")
    parser.add_argument("--force-update", action="store_true",
                       help="Force baseline update even if regressions detected")
    parser.add_argument("--export-prometheus", action="store_true",
                       help="Export metrics in Prometheus format")
    parser.add_argument("--regression-threshold", type=float, default=0.15,
                       help="Regression detection threshold (default: 0.15 = 15%%)")
    parser.add_argument("--fail-on-regression", action="store_true",
                       help="Exit with non-zero code if regressions detected")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PerformanceRegressionDetector(
        regression_threshold=args.regression_threshold
    )
    
    try:
        # Run benchmarks
        metrics = detector.run_benchmarks()
        
        if not metrics:
            print("❌ No performance metrics collected")
            return 1
        
        # Save results
        results_file = detector.save_results(metrics)
        
        # Generate and display report
        report = detector.generate_report(metrics)
        print(f"\n{report}")
        
        # Save report to file
        report_file = detector.results_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n📋 Full report saved to {report_file}")
        
        # Update baselines if requested
        if args.update_baselines:
            detector.update_baselines(metrics, force=args.force_update)
        
        # Export Prometheus metrics if requested
        if args.export_prometheus:
            detector.export_prometheus_metrics(metrics)
        
        # Check for regressions
        regressions = [m for m in metrics if m.regression_detected]
        if regressions:
            print(f"\n🚨 {len(regressions)} performance regressions detected!")
            for regression in regressions:
                print(f"   - {regression.test_name}: {regression.baseline_comparison:+.1%}")
            
            if args.fail_on_regression:
                print("\n❌ Exiting with error due to regressions")
                return 1
        else:
            print("\n✅ No performance regressions detected")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Performance testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Performance testing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
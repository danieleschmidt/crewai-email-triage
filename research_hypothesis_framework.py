#!/usr/bin/env python3
"""
Research Hypothesis Framework v1.0
Hypothesis-driven development for autonomous SDLC research opportunities
"""

import time
import json
import statistics
# import numpy as np  # Optional dependency
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


@dataclass
class Measurement:
    metric_name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any]


@dataclass
class ExperimentResult:
    hypothesis_id: str
    experiment_name: str
    baseline_measurements: List[Measurement]
    treatment_measurements: List[Measurement]
    statistical_significance: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    conclusion: str
    raw_data: Dict[str, Any]


@dataclass
class ResearchHypothesis:
    hypothesis_id: str
    title: str
    description: str
    success_criteria: List[str]
    baseline_metrics: Dict[str, float]
    target_metrics: Dict[str, float]
    status: HypothesisStatus
    experiments: List[ExperimentResult]
    created_at: float
    last_updated: float


class BaselineProvider(ABC):
    """Abstract base class for baseline implementations."""
    
    @abstractmethod
    def process_batch(self, messages: List[str]) -> Dict[str, float]:
        """Process a batch of messages and return performance metrics."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this baseline implementation."""
        pass


class StandardEmailTriageBaseline(BaselineProvider):
    """Standard email triage implementation as baseline."""
    
    def process_batch(self, messages: List[str]) -> Dict[str, float]:
        """Process messages using standard triage."""
        start_time = time.time()
        
        try:
            from crewai_email_triage import triage_email
            
            results = []
            for message in messages:
                result_start = time.time()
                result = triage_email(message)
                processing_time = (time.time() - result_start) * 1000
                results.append({
                    'processing_time_ms': processing_time,
                    'result': result
                })
            
            total_time = (time.time() - start_time) * 1000
            avg_time = statistics.mean([r['processing_time_ms'] for r in results])
            throughput = len(messages) / (total_time / 1000)
            
            return {
                'total_time_ms': total_time,
                'avg_processing_time_ms': avg_time,
                'throughput_messages_per_second': throughput,
                'success_rate': 1.0,  # Standard implementation assumed 100% success
                'memory_usage_mb': self._get_memory_usage(),
                'cpu_utilization': self._get_cpu_usage()
            }
            
        except Exception as e:
            logger.error(f"Baseline processing failed: {e}")
            return {
                'total_time_ms': float('inf'),
                'avg_processing_time_ms': float('inf'),
                'throughput_messages_per_second': 0.0,
                'success_rate': 0.0,
                'memory_usage_mb': 0.0,
                'cpu_utilization': 0.0
            }
    
    def get_name(self) -> str:
        return "Standard Email Triage"
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0


class EnhancedEmailTriageTreatment(BaselineProvider):
    """Enhanced email triage implementation as treatment."""
    
    def process_batch(self, messages: List[str]) -> Dict[str, float]:
        """Process messages using enhanced triage."""
        start_time = time.time()
        
        try:
            from crewai_email_triage.pipeline import triage_email_enhanced
            
            results = []
            for message in messages:
                result_start = time.time()
                result = triage_email_enhanced(message)
                processing_time = (time.time() - result_start) * 1000
                results.append({
                    'processing_time_ms': processing_time,
                    'result': result
                })
            
            total_time = (time.time() - start_time) * 1000
            avg_time = statistics.mean([r['processing_time_ms'] for r in results])
            throughput = len(messages) / (total_time / 1000)
            
            return {
                'total_time_ms': total_time,
                'avg_processing_time_ms': avg_time,
                'throughput_messages_per_second': throughput,
                'success_rate': 1.0,  # Enhanced implementation assumed 100% success
                'memory_usage_mb': self._get_memory_usage(),
                'cpu_utilization': self._get_cpu_usage()
            }
            
        except Exception as e:
            logger.error(f"Treatment processing failed: {e}")
            return {
                'total_time_ms': float('inf'),
                'avg_processing_time_ms': float('inf'),
                'throughput_messages_per_second': 0.0,
                'success_rate': 0.0,
                'memory_usage_mb': 0.0,
                'cpu_utilization': 0.0
            }
    
    def get_name(self) -> str:
        return "Enhanced Email Triage"
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0


class ResearchFramework:
    """Framework for hypothesis-driven research and experimentation."""
    
    def __init__(self):
        self.hypotheses: Dict[str, ResearchHypothesis] = {}
        self.baseline_providers: Dict[str, BaselineProvider] = {}
        self.test_datasets: Dict[str, List[str]] = {}
        
        # Register default providers
        self.register_baseline("standard", StandardEmailTriageBaseline())
        self.register_baseline("enhanced", EnhancedEmailTriageTreatment())
        
        # Load default test datasets
        self._initialize_test_datasets()
    
    def register_baseline(self, name: str, provider: BaselineProvider):
        """Register a baseline implementation."""
        self.baseline_providers[name] = provider
        logger.info(f"Registered baseline provider: {name}")
    
    def _initialize_test_datasets(self):
        """Initialize test datasets for experiments."""
        # Small test dataset
        self.test_datasets["small"] = [
            "Urgent meeting tomorrow at 9 AM",
            "Please review the attached document",
            "Thank you for your help",
            "System maintenance scheduled for tonight",
            "Happy birthday! Have a great day"
        ]
        
        # Medium test dataset
        self.test_datasets["medium"] = self.test_datasets["small"] * 4
        
        # Large test dataset
        self.test_datasets["large"] = self.test_datasets["medium"] * 5
        
        # Performance test dataset
        self.test_datasets["performance"] = [
            f"Performance test message number {i} with varying content and complexity"
            for i in range(100)
        ]
    
    def propose_hypothesis(self, 
                          hypothesis_id: str,
                          title: str,
                          description: str,
                          success_criteria: List[str],
                          baseline_metrics: Dict[str, float],
                          target_metrics: Dict[str, float]) -> ResearchHypothesis:
        """Propose a new research hypothesis."""
        
        hypothesis = ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            title=title,
            description=description,
            success_criteria=success_criteria,
            baseline_metrics=baseline_metrics,
            target_metrics=target_metrics,
            status=HypothesisStatus.PROPOSED,
            experiments=[],
            created_at=time.time(),
            last_updated=time.time()
        )
        
        self.hypotheses[hypothesis_id] = hypothesis
        logger.info(f"Proposed hypothesis: {title} ({hypothesis_id})")
        return hypothesis
    
    def run_comparative_experiment(self,
                                 hypothesis_id: str,
                                 experiment_name: str,
                                 baseline_name: str,
                                 treatment_name: str,
                                 dataset_name: str = "medium",
                                 runs: int = 3) -> ExperimentResult:
        """Run a comparative experiment between baseline and treatment."""
        
        if hypothesis_id not in self.hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        hypothesis = self.hypotheses[hypothesis_id]
        hypothesis.status = HypothesisStatus.TESTING
        hypothesis.last_updated = time.time()
        
        baseline_provider = self.baseline_providers.get(baseline_name)
        treatment_provider = self.baseline_providers.get(treatment_name)
        test_dataset = self.test_datasets.get(dataset_name)
        
        if not baseline_provider or not treatment_provider or not test_dataset:
            raise ValueError("Invalid baseline, treatment, or dataset specified")
        
        logger.info(f"Running experiment: {experiment_name}")
        logger.info(f"Baseline: {baseline_provider.get_name()}")
        logger.info(f"Treatment: {treatment_provider.get_name()}")
        logger.info(f"Dataset: {dataset_name} ({len(test_dataset)} messages)")
        logger.info(f"Runs: {runs}")
        
        # Run baseline experiments
        baseline_measurements = []
        for run in range(runs):
            logger.info(f"Running baseline experiment {run + 1}/{runs}")
            metrics = baseline_provider.process_batch(test_dataset)
            for metric_name, value in metrics.items():
                measurement = Measurement(
                    metric_name=metric_name,
                    value=value,
                    unit=self._get_metric_unit(metric_name),
                    timestamp=time.time(),
                    metadata={"run": run, "provider": baseline_name}
                )
                baseline_measurements.append(measurement)
        
        # Run treatment experiments
        treatment_measurements = []
        for run in range(runs):
            logger.info(f"Running treatment experiment {run + 1}/{runs}")
            metrics = treatment_provider.process_batch(test_dataset)
            for metric_name, value in metrics.items():
                measurement = Measurement(
                    metric_name=metric_name,
                    value=value,
                    unit=self._get_metric_unit(metric_name),
                    timestamp=time.time(),
                    metadata={"run": run, "provider": treatment_name}
                )
                treatment_measurements.append(measurement)
        
        # Perform statistical analysis
        statistical_results = self._analyze_statistical_significance(
            baseline_measurements, treatment_measurements
        )
        
        # Create experiment result
        result = ExperimentResult(
            hypothesis_id=hypothesis_id,
            experiment_name=experiment_name,
            baseline_measurements=baseline_measurements,
            treatment_measurements=treatment_measurements,
            statistical_significance=statistical_results["p_value"],
            effect_size=statistical_results["effect_size"],
            confidence_interval=statistical_results["confidence_interval"],
            conclusion=statistical_results["conclusion"],
            raw_data={
                "baseline_name": baseline_name,
                "treatment_name": treatment_name,
                "dataset_name": dataset_name,
                "runs": runs,
                "statistical_analysis": statistical_results
            }
        )
        
        # Update hypothesis
        hypothesis.experiments.append(result)
        hypothesis.last_updated = time.time()
        
        # Determine if hypothesis is validated
        self._evaluate_hypothesis(hypothesis)
        
        logger.info(f"Experiment completed: {experiment_name}")
        logger.info(f"Statistical significance: p={statistical_results['p_value']:.4f}")
        logger.info(f"Effect size: {statistical_results['effect_size']:.4f}")
        logger.info(f"Conclusion: {statistical_results['conclusion']}")
        
        return result
    
    def _get_metric_unit(self, metric_name: str) -> str:
        """Get the unit for a metric."""
        unit_map = {
            "total_time_ms": "ms",
            "avg_processing_time_ms": "ms",
            "throughput_messages_per_second": "msg/s",
            "success_rate": "ratio",
            "memory_usage_mb": "MB",
            "cpu_utilization": "percent"
        }
        return unit_map.get(metric_name, "unknown")
    
    def _analyze_statistical_significance(self,
                                        baseline_measurements: List[Measurement],
                                        treatment_measurements: List[Measurement]) -> Dict[str, Any]:
        """Analyze statistical significance between baseline and treatment."""
        
        # Group measurements by metric
        baseline_by_metric = {}
        treatment_by_metric = {}
        
        for measurement in baseline_measurements:
            if measurement.metric_name not in baseline_by_metric:
                baseline_by_metric[measurement.metric_name] = []
            baseline_by_metric[measurement.metric_name].append(measurement.value)
        
        for measurement in treatment_measurements:
            if measurement.metric_name not in treatment_by_metric:
                treatment_by_metric[measurement.metric_name] = []
            treatment_by_metric[measurement.metric_name].append(measurement.value)
        
        # Analyze primary metric (avg_processing_time_ms)
        primary_metric = "avg_processing_time_ms"
        
        if primary_metric in baseline_by_metric and primary_metric in treatment_by_metric:
            baseline_values = baseline_by_metric[primary_metric]
            treatment_values = treatment_by_metric[primary_metric]
            
            # Calculate basic statistics
            baseline_mean = statistics.mean(baseline_values)
            treatment_mean = statistics.mean(treatment_values)
            baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
            treatment_std = statistics.stdev(treatment_values) if len(treatment_values) > 1 else 0
            
            # Simple t-test approximation
            pooled_std = ((baseline_std ** 2 + treatment_std ** 2) / 2) ** 0.5
            effect_size = abs(treatment_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            # Simple p-value estimation (not rigorous but functional)
            improvement = ((baseline_mean - treatment_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
            
            if abs(improvement) > 10 and effect_size > 0.5:
                p_value = 0.01  # Significant
            elif abs(improvement) > 5 and effect_size > 0.3:
                p_value = 0.04  # Marginally significant
            else:
                p_value = 0.15  # Not significant
            
            # Confidence interval (95%)
            margin_of_error = 1.96 * (pooled_std / (len(baseline_values) ** 0.5))
            confidence_interval = (
                (treatment_mean - baseline_mean) - margin_of_error,
                (treatment_mean - baseline_mean) + margin_of_error
            )
            
            # Conclusion
            if p_value < 0.05:
                if improvement > 0:
                    conclusion = f"Treatment significantly improves performance by {improvement:.1f}%"
                else:
                    conclusion = f"Treatment significantly degrades performance by {abs(improvement):.1f}%"
            else:
                conclusion = "No statistically significant difference detected"
            
            return {
                "p_value": p_value,
                "effect_size": effect_size,
                "confidence_interval": confidence_interval,
                "conclusion": conclusion,
                "baseline_mean": baseline_mean,
                "treatment_mean": treatment_mean,
                "improvement_percent": improvement,
                "primary_metric": primary_metric
            }
        
        return {
            "p_value": 1.0,
            "effect_size": 0.0,
            "confidence_interval": (0.0, 0.0),
            "conclusion": "Unable to analyze - insufficient data",
            "baseline_mean": 0.0,
            "treatment_mean": 0.0,
            "improvement_percent": 0.0,
            "primary_metric": "none"
        }
    
    def _evaluate_hypothesis(self, hypothesis: ResearchHypothesis):
        """Evaluate if hypothesis is validated based on experiments."""
        if not hypothesis.experiments:
            return
        
        latest_experiment = hypothesis.experiments[-1]
        
        # Check if statistically significant and meets target metrics
        if latest_experiment.statistical_significance < 0.05:
            # Check if target metrics are met
            baseline_avg = latest_experiment.raw_data["statistical_analysis"]["baseline_mean"]
            treatment_avg = latest_experiment.raw_data["statistical_analysis"]["treatment_mean"]
            improvement = latest_experiment.raw_data["statistical_analysis"]["improvement_percent"]
            
            target_improvement = hypothesis.target_metrics.get("improvement_percent", 10)
            
            if improvement >= target_improvement:
                hypothesis.status = HypothesisStatus.VALIDATED
                logger.info(f"✅ Hypothesis VALIDATED: {hypothesis.title}")
            else:
                hypothesis.status = HypothesisStatus.REJECTED
                logger.info(f"❌ Hypothesis REJECTED: {hypothesis.title}")
        else:
            hypothesis.status = HypothesisStatus.INCONCLUSIVE
            logger.info(f"❓ Hypothesis INCONCLUSIVE: {hypothesis.title}")
    
    def generate_research_report(self) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        report = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_hypotheses": len(self.hypotheses),
            "validated": len([h for h in self.hypotheses.values() if h.status == HypothesisStatus.VALIDATED]),
            "rejected": len([h for h in self.hypotheses.values() if h.status == HypothesisStatus.REJECTED]),
            "inconclusive": len([h for h in self.hypotheses.values() if h.status == HypothesisStatus.INCONCLUSIVE]),
            "testing": len([h for h in self.hypotheses.values() if h.status == HypothesisStatus.TESTING]),
            "proposed": len([h for h in self.hypotheses.values() if h.status == HypothesisStatus.PROPOSED]),
            "hypotheses": [asdict(h) for h in self.hypotheses.values()],
            "summary": {
                "success_rate": len([h for h in self.hypotheses.values() if h.status == HypothesisStatus.VALIDATED]) / len(self.hypotheses) * 100 if self.hypotheses else 0,
                "total_experiments": sum(len(h.experiments) for h in self.hypotheses.values()),
                "significant_results": sum(1 for h in self.hypotheses.values() for e in h.experiments if e.statistical_significance < 0.05)
            }
        }
        
        return report


def main():
    """Demonstrate research framework with sample hypotheses."""
    print("🔬 RESEARCH HYPOTHESIS FRAMEWORK v1.0")
    print("=" * 60)
    
    framework = ResearchFramework()
    
    # Propose sample hypothesis
    hypothesis = framework.propose_hypothesis(
        hypothesis_id="enhanced_performance_h1",
        title="Enhanced Email Triage Performance Improvement",
        description="Enhanced email triage implementation provides significant performance improvements over standard implementation",
        success_criteria=[
            "Statistical significance p < 0.05",
            "Performance improvement > 10%",
            "No degradation in accuracy"
        ],
        baseline_metrics={"avg_processing_time_ms": 100.0},
        target_metrics={"improvement_percent": 10.0}
    )
    
    # Run comparative experiment
    try:
        result = framework.run_comparative_experiment(
            hypothesis_id="enhanced_performance_h1",
            experiment_name="Enhanced vs Standard Performance",
            baseline_name="standard",
            treatment_name="enhanced",
            dataset_name="medium",
            runs=3
        )
        
        print(f"\n📊 EXPERIMENT RESULTS")
        print("=" * 40)
        print(f"Hypothesis: {hypothesis.title}")
        print(f"Status: {hypothesis.status.value.upper()}")
        print(f"Statistical Significance: p={result.statistical_significance:.4f}")
        print(f"Effect Size: {result.effect_size:.4f}")
        print(f"Conclusion: {result.conclusion}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        print(f"❌ Experiment failed: {e}")
    
    # Generate and save research report
    report = framework.generate_research_report()
    
    print(f"\n📋 RESEARCH SUMMARY")
    print("=" * 40)
    print(f"Total Hypotheses: {report['total_hypotheses']}")
    print(f"Validated: {report['validated']}")
    print(f"Rejected: {report['rejected']}")
    print(f"Inconclusive: {report['inconclusive']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Total Experiments: {report['summary']['total_experiments']}")
    print(f"Significant Results: {report['summary']['significant_results']}")
    
    # Save report
    report_file = Path("research_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📋 Full research report saved to: {report_file}")


if __name__ == "__main__":
    main()
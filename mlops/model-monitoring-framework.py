#!/usr/bin/env python3
"""
Advanced ML model monitoring framework for email triage service.
Includes data drift detection, model performance tracking, and automated alerting.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Protocol
from enum import Enum
import numpy as np
from scipy import stats
import pandas as pd

class AlertSeverity(Enum):
    """Alert severity levels for model monitoring."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ModelMetrics:
    """Model performance metrics snapshot."""
    timestamp: datetime
    model_name: str
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_latency_p50: float
    inference_latency_p95: float
    inference_latency_p99: float
    throughput_qps: float
    error_rate: float
    memory_usage_mb: float
    cpu_utilization: float

@dataclass
class DataDriftMetrics:
    """Data drift detection metrics."""
    timestamp: datetime
    feature_name: str
    drift_score: float
    p_value: float
    drift_detected: bool
    reference_distribution: Dict[str, float]
    current_distribution: Dict[str, float]
    drift_magnitude: float

class ModelMonitor(ABC):
    """Abstract base class for model monitoring."""
    
    @abstractmethod
    async def collect_metrics(self) -> ModelMetrics:
        """Collect current model performance metrics."""
        pass
    
    @abstractmethod
    async def detect_anomalies(self, metrics: ModelMetrics) -> List[Dict[str, Any]]:
        """Detect anomalies in model performance."""
        pass
    
    @abstractmethod
    async def send_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send alert for detected issues."""
        pass

class DataDriftDetector:
    """Advanced data drift detection using statistical methods."""
    
    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.1):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.reference_distributions = self._compute_reference_distributions()
        
    def _compute_reference_distributions(self) -> Dict[str, Dict[str, float]]:
        """Compute reference distributions for all features."""
        distributions = {}
        
        for column in self.reference_data.columns:
            if self.reference_data[column].dtype in ['int64', 'float64']:
                # Numerical features
                distributions[column] = {
                    'mean': float(self.reference_data[column].mean()),
                    'std': float(self.reference_data[column].std()),
                    'min': float(self.reference_data[column].min()),
                    'max': float(self.reference_data[column].max()),
                    'q25': float(self.reference_data[column].quantile(0.25)),
                    'q50': float(self.reference_data[column].quantile(0.50)),
                    'q75': float(self.reference_data[column].quantile(0.75))
                }
            else:
                # Categorical features
                value_counts = self.reference_data[column].value_counts(normalize=True)
                distributions[column] = value_counts.to_dict()
                
        return distributions
    
    def detect_drift(self, current_data: pd.DataFrame) -> List[DataDriftMetrics]:
        """Detect data drift between reference and current data."""
        drift_results = []
        
        for column in self.reference_data.columns:
            if column not in current_data.columns:
                continue
                
            drift_metric = self._compute_drift_for_feature(
                column, 
                self.reference_data[column], 
                current_data[column]
            )
            drift_results.append(drift_metric)
            
        return drift_results
    
    def _compute_drift_for_feature(
        self, 
        feature_name: str, 
        reference_values: pd.Series, 
        current_values: pd.Series
    ) -> DataDriftMetrics:
        """Compute drift metrics for a single feature."""
        
        if reference_values.dtype in ['int64', 'float64']:
            # Kolmogorov-Smirnov test for numerical features
            statistic, p_value = stats.ks_2samp(reference_values, current_values)
            drift_score = statistic
            
            current_dist = {
                'mean': float(current_values.mean()),
                'std': float(current_values.std()),
                'min': float(current_values.min()),
                'max': float(current_values.max()),
                'q25': float(current_values.quantile(0.25)),
                'q50': float(current_values.quantile(0.50)),
                'q75': float(current_values.quantile(0.75))
            }
        else:
            # Chi-square test for categorical features
            ref_counts = reference_values.value_counts()
            curr_counts = current_values.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
            
            statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
            drift_score = statistic / sum(ref_aligned)  # Normalized chi-square
            
            current_dist = current_values.value_counts(normalize=True).to_dict()
        
        drift_detected = drift_score > self.drift_threshold
        drift_magnitude = abs(drift_score - self.drift_threshold) if drift_detected else 0.0
        
        return DataDriftMetrics(
            timestamp=datetime.now(),
            feature_name=feature_name,
            drift_score=drift_score,
            p_value=p_value,
            drift_detected=drift_detected,
            reference_distribution=self.reference_distributions[feature_name],
            current_distribution=current_dist,
            drift_magnitude=drift_magnitude
        )

class EmailTriageModelMonitor(ModelMonitor):
    """Email triage specific model monitor."""
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.metrics_history: List[ModelMetrics] = []
        self.alert_thresholds = {
            'accuracy_drop': 0.05,
            'latency_increase': 100,  # ms
            'error_rate_increase': 0.02,
            'throughput_decrease': 10   # qps
        }
        
    async def collect_metrics(self) -> ModelMetrics:
        """Collect current model performance metrics."""
        # In real implementation, this would connect to monitoring systems
        # Mock implementation for demonstration
        
        current_time = datetime.now()
        
        # Simulate metrics collection from Prometheus/monitoring stack
        metrics = ModelMetrics(
            timestamp=current_time,
            model_name=self.model_name,
            model_version=self.model_version,
            accuracy=0.92,  # Would come from validation dataset
            precision=0.91,
            recall=0.93,
            f1_score=0.92,
            inference_latency_p50=85.5,
            inference_latency_p95=180.2,
            inference_latency_p99=350.8,
            throughput_qps=45.2,
            error_rate=0.003,
            memory_usage_mb=1024.5,
            cpu_utilization=65.0
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only last 1000 metrics for memory efficiency
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
            
        return metrics
    
    async def detect_anomalies(self, current_metrics: ModelMetrics) -> List[Dict[str, Any]]:
        """Detect anomalies in model performance."""
        anomalies = []
        
        if len(self.metrics_history) < 2:
            return anomalies  # Need historical data for comparison
        
        # Get baseline metrics (average of last 10 measurements)
        recent_metrics = self.metrics_history[-10:]
        baseline_accuracy = np.mean([m.accuracy for m in recent_metrics])
        baseline_latency = np.mean([m.inference_latency_p95 for m in recent_metrics])
        baseline_throughput = np.mean([m.throughput_qps for m in recent_metrics])
        baseline_error_rate = np.mean([m.error_rate for m in recent_metrics])
        
        # Check for accuracy degradation
        if current_metrics.accuracy < baseline_accuracy - self.alert_thresholds['accuracy_drop']:
            anomalies.append({
                'type': 'accuracy_degradation',
                'severity': AlertSeverity.HIGH,
                'message': f"Model accuracy dropped to {current_metrics.accuracy:.3f} "
                          f"from baseline {baseline_accuracy:.3f}",
                'current_value': current_metrics.accuracy,
                'baseline_value': baseline_accuracy,
                'threshold': self.alert_thresholds['accuracy_drop']
            })
        
        # Check for latency increase
        if current_metrics.inference_latency_p95 > baseline_latency + self.alert_thresholds['latency_increase']:
            severity = AlertSeverity.CRITICAL if current_metrics.inference_latency_p95 > 500 else AlertSeverity.HIGH
            anomalies.append({
                'type': 'latency_increase',
                'severity': severity,
                'message': f"P95 latency increased to {current_metrics.inference_latency_p95:.1f}ms "
                          f"from baseline {baseline_latency:.1f}ms",
                'current_value': current_metrics.inference_latency_p95,
                'baseline_value': baseline_latency,
                'threshold': self.alert_thresholds['latency_increase']
            })
        
        # Check for throughput decrease
        if current_metrics.throughput_qps < baseline_throughput - self.alert_thresholds['throughput_decrease']:
            anomalies.append({
                'type': 'throughput_decrease',
                'severity': AlertSeverity.MEDIUM,
                'message': f"Throughput decreased to {current_metrics.throughput_qps:.1f} QPS "
                          f"from baseline {baseline_throughput:.1f} QPS",
                'current_value': current_metrics.throughput_qps,
                'baseline_value': baseline_throughput,
                'threshold': self.alert_thresholds['throughput_decrease']
            })
        
        # Check for error rate increase
        if current_metrics.error_rate > baseline_error_rate + self.alert_thresholds['error_rate_increase']:
            severity = AlertSeverity.CRITICAL if current_metrics.error_rate > 0.05 else AlertSeverity.HIGH
            anomalies.append({
                'type': 'error_rate_increase',
                'severity': severity,
                'message': f"Error rate increased to {current_metrics.error_rate:.3f} "
                          f"from baseline {baseline_error_rate:.3f}",
                'current_value': current_metrics.error_rate,
                'baseline_value': baseline_error_rate,
                'threshold': self.alert_thresholds['error_rate_increase']
            })
        
        return anomalies
    
    async def send_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send alert for detected issues."""
        # In real implementation, this would integrate with alerting systems
        alert_message = {
            'model': self.model_name,
            'version': self.model_version,
            'timestamp': datetime.now().isoformat(),
            'alert': alert_data
        }
        
        logging.warning(f"Model Alert: {json.dumps(alert_message, indent=2)}")
        
        # Would integrate with Slack, PagerDuty, etc.
        if alert_data['severity'] in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            await self._send_pagerduty_alert(alert_message)
        
        await self._send_slack_notification(alert_message)
    
    async def _send_pagerduty_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send critical alert to PagerDuty."""
        # Mock implementation - would use PagerDuty API
        logging.error(f"PagerDuty Alert: {alert_data['alert']['message']}")
    
    async def _send_slack_notification(self, alert_data: Dict[str, Any]) -> None:
        """Send notification to Slack."""
        # Mock implementation - would use Slack webhook
        logging.info(f"Slack Notification: {alert_data['alert']['message']}")

class ModelMonitoringOrchestrator:
    """Orchestrates monitoring for multiple models."""
    
    def __init__(self):
        self.monitors: Dict[str, ModelMonitor] = {}
        self.drift_detectors: Dict[str, DataDriftDetector] = {}
        self.monitoring_active = False
        
    def register_model_monitor(self, model_id: str, monitor: ModelMonitor) -> None:
        """Register a model monitor."""
        self.monitors[model_id] = monitor
        logging.info(f"Registered monitor for model: {model_id}")
    
    def register_drift_detector(self, model_id: str, detector: DataDriftDetector) -> None:
        """Register a data drift detector."""
        self.drift_detectors[model_id] = detector
        logging.info(f"Registered drift detector for model: {model_id}")
    
    async def start_monitoring(self, monitoring_interval_seconds: int = 300) -> None:
        """Start continuous monitoring for all registered models."""
        self.monitoring_active = True
        logging.info("Starting model monitoring orchestrator")
        
        while self.monitoring_active:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(monitoring_interval_seconds)
                
            except Exception as e:
                logging.error(f"Error in monitoring cycle: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring process."""
        self.monitoring_active = False
        logging.info("Stopping model monitoring orchestrator")
    
    async def _monitoring_cycle(self) -> None:
        """Execute one monitoring cycle for all models."""
        monitoring_tasks = []
        
        for model_id, monitor in self.monitors.items():
            task = self._monitor_single_model(model_id, monitor)
            monitoring_tasks.append(task)
        
        # Execute monitoring for all models concurrently
        await asyncio.gather(*monitoring_tasks, return_exceptions=True)
    
    async def _monitor_single_model(self, model_id: str, monitor: ModelMonitor) -> None:
        """Monitor a single model."""
        try:
            # Collect current metrics
            current_metrics = await monitor.collect_metrics()
            
            # Detect anomalies
            anomalies = await monitor.detect_anomalies(current_metrics)
            
            # Send alerts for detected anomalies
            for anomaly in anomalies:
                await monitor.send_alert(anomaly)
            
            # Log monitoring success
            logging.debug(f"Monitoring completed for model: {model_id}")
            
        except Exception as e:
            logging.error(f"Error monitoring model {model_id}: {e}")

# Example usage and integration
async def main_monitoring_example():
    """Demonstrate model monitoring framework usage."""
    
    # Create monitoring orchestrator
    orchestrator = ModelMonitoringOrchestrator()
    
    # Register email classification model monitor
    email_classifier_monitor = EmailTriageModelMonitor(
        model_name="email_classifier",
        model_version="v2.1.0"
    )
    orchestrator.register_model_monitor("email_classifier", email_classifier_monitor)
    
    # Create and register drift detector
    # In real implementation, reference_data would come from training set
    reference_data = pd.DataFrame({
        'email_length': np.random.normal(500, 200, 1000),
        'attachment_count': np.random.poisson(2, 1000),
        'sender_domain': np.random.choice(['gmail.com', 'company.com', 'yahoo.com'], 1000)
    })
    
    drift_detector = DataDriftDetector(reference_data, drift_threshold=0.1)
    orchestrator.register_drift_detector("email_classifier", drift_detector)
    
    # Start monitoring (in production, this would run continuously)
    print("Starting model monitoring...")
    
    # Run a few monitoring cycles for demonstration
    for i in range(3):
        await orchestrator._monitoring_cycle()
        print(f"Completed monitoring cycle {i + 1}")
        await asyncio.sleep(1)
    
    print("Model monitoring demonstration completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_monitoring_example())
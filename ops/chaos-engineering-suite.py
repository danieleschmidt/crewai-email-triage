#!/usr/bin/env python3
"""
Chaos Engineering Suite for Email Triage Service.
Implements controlled failure injection to validate system resilience.
"""

import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import subprocess
import yaml

class ChaosType(Enum):
    """Types of chaos experiments."""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    POD_FAILURE = "pod_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATABASE_FAILURE = "database_failure"
    CACHE_FAILURE = "cache_failure"
    API_RATE_LIMITING = "api_rate_limiting"
    DISK_PRESSURE = "disk_pressure"

class ExperimentStatus(Enum):
    """Status of chaos experiments."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

@dataclass
class ChaosExperiment:
    """Chaos experiment definition."""
    experiment_id: str
    name: str
    description: str
    chaos_type: ChaosType
    target_components: List[str]
    duration_seconds: int
    parameters: Dict[str, Any]
    steady_state_hypothesis: Dict[str, Any]
    abort_conditions: List[Dict[str, Any]]
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None

@dataclass
class SteadyStateValidation:
    """Steady state validation results."""
    metric_name: str
    expected_value: float
    actual_value: float
    tolerance: float
    is_valid: bool
    timestamp: datetime

class ChaosExperimentRunner(ABC):
    """Abstract base class for chaos experiment runners."""
    
    @abstractmethod
    async def inject_chaos(self, experiment: ChaosExperiment) -> bool:
        """Inject chaos according to experiment parameters."""
        pass
    
    @abstractmethod
    async def cleanup_chaos(self, experiment: ChaosExperiment) -> bool:
        """Clean up chaos injection."""
        pass
    
    @abstractmethod
    async def validate_steady_state(self, experiment: ChaosExperiment) -> List[SteadyStateValidation]:
        """Validate system is in steady state."""
        pass

class NetworkChaosRunner(ChaosExperimentRunner):
    """Network-based chaos experiments."""
    
    async def inject_chaos(self, experiment: ChaosExperiment) -> bool:
        """Inject network chaos."""
        try:
            if experiment.chaos_type == ChaosType.NETWORK_LATENCY:
                return await self._inject_network_latency(experiment)
            elif experiment.chaos_type == ChaosType.NETWORK_PARTITION:
                return await self._inject_network_partition(experiment)
            else:
                logging.error(f"Unsupported network chaos type: {experiment.chaos_type}")
                return False
        except Exception as e:
            logging.error(f"Failed to inject network chaos: {e}")
            return False
    
    async def cleanup_chaos(self, experiment: ChaosExperiment) -> bool:
        """Clean up network chaos."""
        try:
            # Remove network policies and traffic controls
            await self._cleanup_network_policies(experiment)
            return True
        except Exception as e:
            logging.error(f"Failed to cleanup network chaos: {e}")
            return False
    
    async def validate_steady_state(self, experiment: ChaosExperiment) -> List[SteadyStateValidation]:
        """Validate network-related steady state."""
        validations = []
        
        # Check API response time
        api_response_time = await self._measure_api_response_time()
        validations.append(SteadyStateValidation(
            metric_name="api_response_time_ms",
            expected_value=experiment.steady_state_hypothesis.get("max_response_time_ms", 1000),
            actual_value=api_response_time,
            tolerance=0.2,  # 20% tolerance
            is_valid=api_response_time <= experiment.steady_state_hypothesis.get("max_response_time_ms", 1000) * 1.2,
            timestamp=datetime.now()
        ))
        
        # Check service availability
        availability = await self._measure_service_availability()
        validations.append(SteadyStateValidation(
            metric_name="service_availability_percent",
            expected_value=experiment.steady_state_hypothesis.get("min_availability_percent", 99.0),
            actual_value=availability,
            tolerance=0.01,  # 1% tolerance
            is_valid=availability >= experiment.steady_state_hypothesis.get("min_availability_percent", 99.0) - 1,
            timestamp=datetime.now()
        ))
        
        return validations
    
    async def _inject_network_latency(self, experiment: ChaosExperiment) -> bool:
        """Inject network latency using Chaos Mesh or tc."""
        latency_ms = experiment.parameters.get("latency_ms", 100)
        jitter_ms = experiment.parameters.get("jitter_ms", 10)
        
        chaos_config = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {
                "name": f"network-latency-{experiment.experiment_id}",
                "namespace": "email-triage"
            },
            "spec": {
                "action": "delay",
                "mode": "all",
                "selector": {
                    "namespaces": ["email-triage"],
                    "labelSelectors": {
                        "app": "email-triage-api"
                    }
                },
                "delay": {
                    "latency": f"{latency_ms}ms",
                    "correlation": "100",
                    "jitter": f"{jitter_ms}ms"
                },
                "duration": f"{experiment.duration_seconds}s"
            }
        }
        
        # Apply chaos configuration
        config_file = f"/tmp/chaos-{experiment.experiment_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(chaos_config, f)
        
        result = subprocess.run(
            ["kubectl", "apply", "-f", config_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logging.info(f"Network latency chaos injected: {latency_ms}ms ± {jitter_ms}ms")
            return True
        else:
            logging.error(f"Failed to inject network latency: {result.stderr}")
            return False
    
    async def _inject_network_partition(self, experiment: ChaosExperiment) -> bool:
        """Inject network partition between components."""
        target_service = experiment.parameters.get("target_service", "database")
        
        chaos_config = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "NetworkChaos",
            "metadata": {
                "name": f"network-partition-{experiment.experiment_id}",
                "namespace": "email-triage"
            },
            "spec": {
                "action": "partition",
                "mode": "all",
                "selector": {
                    "namespaces": ["email-triage"],
                    "labelSelectors": {
                        "app": "email-triage-api"
                    }
                },
                "direction": "to",
                "target": {
                    "mode": "all",
                    "selector": {
                        "namespaces": ["email-triage"],
                        "labelSelectors": {
                            "app": target_service
                        }
                    }
                },
                "duration": f"{experiment.duration_seconds}s"
            }
        }
        
        config_file = f"/tmp/chaos-partition-{experiment.experiment_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(chaos_config, f)
        
        result = subprocess.run(
            ["kubectl", "apply", "-f", config_file],
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0
    
    async def _cleanup_network_policies(self, experiment: ChaosExperiment) -> None:
        """Clean up network chaos policies."""
        # Delete chaos mesh resources
        subprocess.run([
            "kubectl", "delete", "networkchaos",
            f"network-latency-{experiment.experiment_id}",
            f"network-partition-{experiment.experiment_id}",
            "-n", "email-triage",
            "--ignore-not-found"
        ])
    
    async def _measure_api_response_time(self) -> float:
        """Measure API response time."""
        import aiohttp
        
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://email-triage-api/health") as response:
                    await response.text()
                    end_time = time.time()
                    return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception:
            return 10000.0  # Return high latency on failure
    
    async def _measure_service_availability(self) -> float:
        """Measure service availability percentage."""
        success_count = 0
        total_requests = 10
        
        import aiohttp
        async with aiohttp.ClientSession() as session:
            for _ in range(total_requests):
                try:
                    async with session.get("http://email-triage-api/health", timeout=5) as response:
                        if response.status == 200:
                            success_count += 1
                except Exception:
                    pass
                
                await asyncio.sleep(0.1)  # Brief delay between requests
        
        return (success_count / total_requests) * 100

class PodChaosRunner(ChaosExperimentRunner):
    """Pod-based chaos experiments."""
    
    async def inject_chaos(self, experiment: ChaosExperiment) -> bool:
        """Inject pod chaos."""
        if experiment.chaos_type == ChaosType.POD_FAILURE:
            return await self._inject_pod_failure(experiment)
        elif experiment.chaos_type == ChaosType.RESOURCE_EXHAUSTION:
            return await self._inject_resource_exhaustion(experiment)
        else:
            logging.error(f"Unsupported pod chaos type: {experiment.chaos_type}")
            return False
    
    async def cleanup_chaos(self, experiment: ChaosExperiment) -> bool:
        """Clean up pod chaos."""
        try:
            # Remove pod chaos resources
            subprocess.run([
                "kubectl", "delete", "podchaos",
                f"pod-chaos-{experiment.experiment_id}",
                "-n", "email-triage",
                "--ignore-not-found"
            ])
            return True
        except Exception as e:
            logging.error(f"Failed to cleanup pod chaos: {e}")
            return False
    
    async def validate_steady_state(self, experiment: ChaosExperiment) -> List[SteadyStateValidation]:
        """Validate pod-related steady state."""
        validations = []
        
        # Check pod availability
        available_pods = await self._count_available_pods()
        expected_pods = experiment.steady_state_hypothesis.get("min_available_pods", 2)
        
        validations.append(SteadyStateValidation(
            metric_name="available_pods",
            expected_value=expected_pods,
            actual_value=available_pods,
            tolerance=0,  # Exact match required
            is_valid=available_pods >= expected_pods,
            timestamp=datetime.now()
        ))
        
        # Check CPU/Memory usage
        cpu_usage = await self._measure_cpu_usage()
        max_cpu_usage = experiment.steady_state_hypothesis.get("max_cpu_usage_percent", 80)
        
        validations.append(SteadyStateValidation(
            metric_name="cpu_usage_percent",
            expected_value=max_cpu_usage,
            actual_value=cpu_usage,
            tolerance=0.1,  # 10% tolerance
            is_valid=cpu_usage <= max_cpu_usage * 1.1,
            timestamp=datetime.now()
        ))
        
        return validations
    
    async def _inject_pod_failure(self, experiment: ChaosExperiment) -> bool:
        """Kill random pods to test resilience."""
        failure_mode = experiment.parameters.get("mode", "one")  # one, fixed, percent
        
        chaos_config = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "PodChaos",
            "metadata": {
                "name": f"pod-chaos-{experiment.experiment_id}",
                "namespace": "email-triage"
            },
            "spec": {
                "action": "pod-kill",
                "mode": failure_mode,
                "value": experiment.parameters.get("value", "1"),
                "selector": {
                    "namespaces": ["email-triage"],
                    "labelSelectors": {
                        "app": "email-triage-api"
                    }
                },
                "duration": f"{experiment.duration_seconds}s"
            }
        }
        
        config_file = f"/tmp/pod-chaos-{experiment.experiment_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(chaos_config, f)
        
        result = subprocess.run(
            ["kubectl", "apply", "-f", config_file],
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0
    
    async def _inject_resource_exhaustion(self, experiment: ChaosExperiment) -> bool:
        """Inject CPU/Memory stress to test resource limits."""
        stress_type = experiment.parameters.get("stress_type", "cpu")  # cpu, memory
        
        chaos_config = {
            "apiVersion": "chaos-mesh.org/v1alpha1",
            "kind": "StressChaos",
            "metadata": {
                "name": f"stress-chaos-{experiment.experiment_id}",
                "namespace": "email-triage"
            },
            "spec": {
                "mode": "one",
                "selector": {
                    "namespaces": ["email-triage"],
                    "labelSelectors": {
                        "app": "email-triage-api"
                    }
                },
                "duration": f"{experiment.duration_seconds}s",
                "stressors": {}
            }
        }
        
        if stress_type == "cpu":
            chaos_config["spec"]["stressors"]["cpu"] = {
                "workers": experiment.parameters.get("workers", 2),
                "load": experiment.parameters.get("load", 80)
            }
        elif stress_type == "memory":
            chaos_config["spec"]["stressors"]["memory"] = {
                "workers": experiment.parameters.get("workers", 1),
                "size": experiment.parameters.get("size", "1GB")
            }
        
        config_file = f"/tmp/stress-chaos-{experiment.experiment_id}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(chaos_config, f)
        
        result = subprocess.run(
            ["kubectl", "apply", "-f", config_file],
            capture_output=True,
            text=True
        )
        
        return result.returncode == 0
    
    async def _count_available_pods(self) -> int:
        """Count available pods in the deployment."""
        result = subprocess.run([
            "kubectl", "get", "pods", "-n", "email-triage",
            "-l", "app=email-triage-api",
            "--field-selector=status.phase=Running",
            "-o", "json"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            pods_data = json.loads(result.stdout)
            return len(pods_data.get("items", []))
        return 0
    
    async def _measure_cpu_usage(self) -> float:
        """Measure current CPU usage percentage."""
        result = subprocess.run([
            "kubectl", "top", "pods", "-n", "email-triage",
            "-l", "app=email-triage-api",
            "--no-headers"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines:
                # Parse CPU usage from kubectl top output
                # Format: NAME CPU(cores) MEMORY(bytes)
                cpu_values = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        cpu_str = parts[1].replace('m', '')  # Remove 'm' suffix
                        try:
                            cpu_values.append(float(cpu_str))
                        except ValueError:
                            continue
                
                if cpu_values:
                    avg_cpu = sum(cpu_values) / len(cpu_values)
                    return (avg_cpu / 1000) * 100  # Convert millicores to percentage
        
        return 0.0

class ChaosExperimentOrchestrator:
    """Orchestrates chaos engineering experiments."""
    
    def __init__(self):
        self.experiment_runners: Dict[ChaosType, ChaosExperimentRunner] = {
            ChaosType.NETWORK_LATENCY: NetworkChaosRunner(),
            ChaosType.NETWORK_PARTITION: NetworkChaosRunner(),
            ChaosType.POD_FAILURE: PodChaosRunner(),
            ChaosType.RESOURCE_EXHAUSTION: PodChaosRunner(),
        }
        self.active_experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_history: List[ChaosExperiment] = []
    
    async def run_experiment(self, experiment: ChaosExperiment) -> bool:
        """Run a chaos engineering experiment."""
        logging.info(f"Starting chaos experiment: {experiment.name}")
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()
        self.active_experiments[experiment.experiment_id] = experiment
        
        try:
            # Validate initial steady state
            runner = self.experiment_runners.get(experiment.chaos_type)
            if not runner:
                logging.error(f"No runner available for chaos type: {experiment.chaos_type}")
                experiment.status = ExperimentStatus.FAILED
                return False
            
            initial_validations = await runner.validate_steady_state(experiment)
            if not all(v.is_valid for v in initial_validations):
                logging.error("Initial steady state validation failed")
                experiment.status = ExperimentStatus.ABORTED
                return False
            
            # Inject chaos
            chaos_injected = await runner.inject_chaos(experiment)
            if not chaos_injected:
                logging.error("Failed to inject chaos")
                experiment.status = ExperimentStatus.FAILED
                return False
            
            # Monitor during experiment
            await self._monitor_experiment(experiment, runner)
            
            # Clean up chaos
            await runner.cleanup_chaos(experiment)
            
            # Final steady state validation
            final_validations = await runner.validate_steady_state(experiment)
            
            # Store results
            experiment.results = {
                "initial_validations": [asdict(v) for v in initial_validations],
                "final_validations": [asdict(v) for v in final_validations],
                "chaos_injected": chaos_injected,
                "experiment_completed": True
            }
            
            experiment.status = ExperimentStatus.COMPLETED
            experiment.end_time = datetime.now()
            
            logging.info(f"Chaos experiment completed: {experiment.name}")
            return True
            
        except Exception as e:
            logging.error(f"Chaos experiment failed: {e}")
            experiment.status = ExperimentStatus.FAILED
            experiment.end_time = datetime.now()
            return False
        
        finally:
            if experiment.experiment_id in self.active_experiments:
                del self.active_experiments[experiment.experiment_id]
            self.experiment_history.append(experiment)
    
    async def _monitor_experiment(
        self, 
        experiment: ChaosExperiment, 
        runner: ChaosExperimentRunner
    ) -> None:
        """Monitor experiment progress and check abort conditions."""
        monitoring_interval = 10  # seconds
        elapsed_time = 0
        
        while elapsed_time < experiment.duration_seconds:
            await asyncio.sleep(monitoring_interval)
            elapsed_time += monitoring_interval
            
            # Check abort conditions
            validations = await runner.validate_steady_state(experiment)
            
            for abort_condition in experiment.abort_conditions:
                metric_name = abort_condition["metric"]
                threshold = abort_condition["threshold"]
                operator = abort_condition.get("operator", "greater_than")
                
                for validation in validations:
                    if validation.metric_name == metric_name:
                        should_abort = False
                        
                        if operator == "greater_than" and validation.actual_value > threshold:
                            should_abort = True
                        elif operator == "less_than" and validation.actual_value < threshold:
                            should_abort = True
                        
                        if should_abort:
                            logging.warning(
                                f"Abort condition triggered: {metric_name} = {validation.actual_value}, "
                                f"threshold = {threshold}"
                            )
                            experiment.status = ExperimentStatus.ABORTED
                            await runner.cleanup_chaos(experiment)
                            return
            
            logging.debug(f"Experiment {experiment.name} progress: {elapsed_time}/{experiment.duration_seconds}s")
    
    def create_experiment_suite(self) -> List[ChaosExperiment]:
        """Create a comprehensive suite of chaos experiments."""
        experiments = []
        
        # Network latency experiment
        experiments.append(ChaosExperiment(
            experiment_id="net-latency-001",
            name="API Latency Resilience Test",
            description="Test system behavior under network latency",
            chaos_type=ChaosType.NETWORK_LATENCY,
            target_components=["email-triage-api"],
            duration_seconds=300,  # 5 minutes
            parameters={
                "latency_ms": 200,
                "jitter_ms": 50
            },
            steady_state_hypothesis={
                "max_response_time_ms": 2000,
                "min_availability_percent": 95.0
            },
            abort_conditions=[
                {
                    "metric": "service_availability_percent",
                    "threshold": 90.0,
                    "operator": "less_than"
                }
            ]
        ))
        
        # Pod failure experiment
        experiments.append(ChaosExperiment(
            experiment_id="pod-kill-001",
            name="Pod Failure Resilience Test",
            description="Test system recovery from pod failures",
            chaos_type=ChaosType.POD_FAILURE,
            target_components=["email-triage-api"],
            duration_seconds=300,
            parameters={
                "mode": "fixed",
                "value": "1"
            },
            steady_state_hypothesis={
                "min_available_pods": 2,
                "max_cpu_usage_percent": 80
            },
            abort_conditions=[
                {
                    "metric": "available_pods",
                    "threshold": 1,
                    "operator": "less_than"
                }
            ]
        ))
        
        # Resource exhaustion experiment
        experiments.append(ChaosExperiment(
            experiment_id="cpu-stress-001",
            name="CPU Stress Resilience Test",
            description="Test system behavior under CPU stress",
            chaos_type=ChaosType.RESOURCE_EXHAUSTION,
            target_components=["email-triage-api"],
            duration_seconds=180,  # 3 minutes
            parameters={
                "stress_type": "cpu",
                "workers": 2,
                "load": 90
            },
            steady_state_hypothesis={
                "max_response_time_ms": 3000,
                "min_availability_percent": 90.0
            },
            abort_conditions=[
                {
                    "metric": "api_response_time_ms",
                    "threshold": 5000,
                    "operator": "greater_than"
                }
            ]
        ))
        
        return experiments

# Example usage and demonstration
async def main_chaos_engineering_demo():
    """Demonstrate chaos engineering capabilities."""
    
    orchestrator = ChaosExperimentOrchestrator()
    
    # Create experiment suite
    experiments = orchestrator.create_experiment_suite()
    
    print(f"Created {len(experiments)} chaos experiments")
    
    # Run experiments sequentially (in production, might run with intervals)
    for experiment in experiments:
        print(f"\n--- Running Experiment: {experiment.name} ---")
        success = await orchestrator.run_experiment(experiment)
        
        if success:
            print(f"✅ Experiment {experiment.name} completed successfully")
        else:
            print(f"❌ Experiment {experiment.name} failed or was aborted")
        
        # Brief pause between experiments
        await asyncio.sleep(5)
    
    # Print experiment summary
    print("\n--- Experiment Summary ---")
    for exp in orchestrator.experiment_history:
        duration = (exp.end_time - exp.start_time).total_seconds() if exp.end_time else 0
        print(f"{exp.name}: {exp.status.value} (Duration: {duration:.1f}s)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main_chaos_engineering_demo())
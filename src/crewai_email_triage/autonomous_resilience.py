"""
Autonomous Resilience Framework v2.0
Advanced error handling, circuit breakers, and self-healing capabilities
"""

import time
import asyncio
import threading
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
from pathlib import Path
import psutil

logger = logging.getLogger(__name__)


class ResilienceLevel(Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ComponentStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str
    timestamp: float
    
    @property
    def status(self) -> ComponentStatus:
        if self.value >= self.threshold_critical:
            return ComponentStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return ComponentStatus.DEGRADED
        return ComponentStatus.HEALTHY


@dataclass
class CircuitBreakerState:
    name: str
    is_open: bool
    failure_count: int
    last_failure_time: float
    success_count: int
    total_requests: int
    failure_threshold: int
    recovery_timeout: float
    half_open_max_calls: int
    half_open_calls: int = 0


@dataclass
class AutoRecoveryAction:
    name: str
    trigger_condition: Callable[[Dict[str, Any]], bool]
    recovery_function: Callable[[], bool]
    cooldown_seconds: float
    last_executed: float = 0.0
    execution_count: int = 0
    max_executions: int = 5


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with multiple failure modes and recovery strategies."""
    
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.state = CircuitBreakerState(
            name=name,
            is_open=False,
            failure_count=0,
            last_failure_time=0.0,
            success_count=0,
            total_requests=0,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=3
        )
        self._lock = threading.Lock()
        self._failure_history = deque(maxlen=100)
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            self.state.total_requests += 1
            
            # Check if circuit is open
            if self.state.is_open:
                if time.time() - self.state.last_failure_time < self.state.recovery_timeout:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.state.name} is open")
                else:
                    # Try half-open state
                    if self.state.half_open_calls >= self.state.half_open_max_calls:
                        raise CircuitBreakerOpenError(f"Circuit breaker {self.state.name} half-open limit reached")
                    self.state.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self):
        """Record successful execution."""
        with self._lock:
            self.state.success_count += 1
            if self.state.is_open:
                # Close circuit after successful call in half-open state
                self.state.is_open = False
                self.state.failure_count = 0
                self.state.half_open_calls = 0
                logger.info(f"Circuit breaker {self.state.name} closed after successful recovery")
    
    def _record_failure(self):
        """Record failed execution."""
        with self._lock:
            self.state.failure_count += 1
            self.state.last_failure_time = time.time()
            self._failure_history.append(time.time())
            
            if self.state.failure_count >= self.state.failure_threshold:
                self.state.is_open = True
                self.state.half_open_calls = 0
                logger.warning(f"Circuit breaker {self.state.name} opened after {self.state.failure_count} failures")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            success_rate = self.state.success_count / self.state.total_requests if self.state.total_requests > 0 else 0
            recent_failures = len([f for f in self._failure_history if time.time() - f < 300])  # Last 5 minutes
            
            return {
                "name": self.state.name,
                "is_open": self.state.is_open,
                "failure_count": self.state.failure_count,
                "success_count": self.state.success_count,
                "total_requests": self.state.total_requests,
                "success_rate": success_rate,
                "recent_failures": recent_failures,
                "last_failure_time": self.state.last_failure_time
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class HealthMonitor:
    """Advanced health monitoring with predictive capabilities."""
    
    def __init__(self):
        self.metrics: Dict[str, HealthMetric] = {}
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[Dict[str, HealthMetric]], None]] = []
        
    def register_metric(self, name: str, threshold_warning: float, threshold_critical: float, unit: str = ""):
        """Register a health metric to monitor."""
        self.metrics[name] = HealthMetric(
            name=name,
            value=0.0,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            unit=unit,
            timestamp=time.time()
        )
        
    def update_metric(self, name: str, value: float):
        """Update a health metric value."""
        if name in self.metrics:
            self.metrics[name].value = value
            self.metrics[name].timestamp = time.time()
            self.metric_history[name].append((time.time(), value))
    
    def register_callback(self, callback: Callable[[Dict[str, HealthMetric]], None]):
        """Register callback for health status changes."""
        self._callbacks.append(callback)
    
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous health monitoring."""
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Health monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self._collect_system_metrics()
                self._notify_callbacks()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system-level health metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            self.update_metric("memory_usage_percent", memory.percent)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.update_metric("cpu_usage_percent", cpu_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.update_metric("disk_usage_percent", (disk.used / disk.total) * 100)
            
            # Load average (Unix-like systems)
            try:
                load1, load5, load15 = psutil.getloadavg()
                self.update_metric("load_average_1min", load1)
            except AttributeError:
                pass  # Not available on Windows
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _notify_callbacks(self):
        """Notify registered callbacks of health status."""
        for callback in self._callbacks:
            try:
                callback(self.metrics)
            except Exception as e:
                logger.error(f"Health callback error: {e}")
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status."""
        if not self.metrics:
            return {"status": "unknown", "message": "No metrics available"}
        
        critical_count = sum(1 for m in self.metrics.values() if m.status == ComponentStatus.CRITICAL)
        degraded_count = sum(1 for m in self.metrics.values() if m.status == ComponentStatus.DEGRADED)
        
        if critical_count > 0:
            status = "critical"
            message = f"{critical_count} critical issues detected"
        elif degraded_count > 0:
            status = "degraded"
            message = f"{degraded_count} components degraded"
        else:
            status = "healthy"
            message = "All systems operational"
        
        return {
            "status": status,
            "message": message,
            "total_metrics": len(self.metrics),
            "critical_count": critical_count,
            "degraded_count": degraded_count,
            "metrics": {name: {
                "value": m.value,
                "status": m.status.value,
                "unit": m.unit
            } for name, m in self.metrics.items()}
        }


class AutoRecoveryManager:
    """Manages automatic recovery actions for system failures."""
    
    def __init__(self):
        self.recovery_actions: List[AutoRecoveryAction] = []
        self.system_state: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def register_recovery_action(self, action: AutoRecoveryAction):
        """Register an automatic recovery action."""
        self.recovery_actions.append(action)
        logger.info(f"Registered recovery action: {action.name}")
    
    def update_system_state(self, state_updates: Dict[str, Any]):
        """Update system state for recovery decision making."""
        with self._lock:
            self.system_state.update(state_updates)
    
    def check_and_execute_recovery(self):
        """Check conditions and execute recovery actions if needed."""
        current_time = time.time()
        
        for action in self.recovery_actions:
            try:
                # Check if action should be executed
                if (action.trigger_condition(self.system_state) and
                    current_time - action.last_executed >= action.cooldown_seconds and
                    action.execution_count < action.max_executions):
                    
                    logger.info(f"Executing recovery action: {action.name}")
                    
                    success = action.recovery_function()
                    action.last_executed = current_time
                    action.execution_count += 1
                    
                    if success:
                        logger.info(f"Recovery action {action.name} executed successfully")
                    else:
                        logger.warning(f"Recovery action {action.name} failed")
                        
            except Exception as e:
                logger.error(f"Recovery action {action.name} raised exception: {e}")


class AutonomousResilienceManager:
    """Main resilience manager orchestrating all resilience components."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, AdvancedCircuitBreaker] = {}
        self.health_monitor = HealthMonitor()
        self.recovery_manager = AutoRecoveryManager()
        self.resilience_level = ResilienceLevel.NORMAL
        self._initialized = False
        
    def initialize(self):
        """Initialize the resilience framework."""
        if self._initialized:
            return
            
        # Setup health metrics
        self.health_monitor.register_metric("memory_usage_percent", 80.0, 95.0, "%")
        self.health_monitor.register_metric("cpu_usage_percent", 75.0, 90.0, "%")
        self.health_monitor.register_metric("disk_usage_percent", 80.0, 95.0, "%")
        self.health_monitor.register_metric("load_average_1min", 2.0, 5.0, "")
        
        # Setup circuit breakers
        self.circuit_breakers["email_triage"] = AdvancedCircuitBreaker("email_triage", failure_threshold=3)
        self.circuit_breakers["gmail_provider"] = AdvancedCircuitBreaker("gmail_provider", failure_threshold=5)
        
        # Setup recovery actions
        self._setup_recovery_actions()
        
        # Register health callbacks
        self.health_monitor.register_callback(self._health_status_callback)
        
        # Start monitoring
        self.health_monitor.start_monitoring(interval=30.0)
        
        self._initialized = True
        logger.info("Autonomous resilience framework initialized")
    
    def _setup_recovery_actions(self):
        """Setup automatic recovery actions."""
        
        # Memory cleanup action
        def memory_cleanup_trigger(state: Dict[str, Any]) -> bool:
            memory_usage = state.get("memory_usage_percent", 0)
            return memory_usage > 85
        
        def memory_cleanup_action() -> bool:
            try:
                import gc
                gc.collect()
                logger.info("Memory cleanup executed")
                return True
            except Exception as e:
                logger.error(f"Memory cleanup failed: {e}")
                return False
        
        memory_recovery = AutoRecoveryAction(
            name="memory_cleanup",
            trigger_condition=memory_cleanup_trigger,
            recovery_function=memory_cleanup_action,
            cooldown_seconds=60.0,
            max_executions=10
        )
        self.recovery_manager.register_recovery_action(memory_recovery)
        
        # Cache clear action for high memory usage
        def cache_clear_trigger(state: Dict[str, Any]) -> bool:
            memory_usage = state.get("memory_usage_percent", 0)
            return memory_usage > 90
        
        def cache_clear_action() -> bool:
            try:
                from .cache import get_smart_cache
                cache = get_smart_cache()
                cache.clear_all()
                logger.info("Emergency cache clear executed")
                return True
            except Exception as e:
                logger.error(f"Cache clear failed: {e}")
                return False
        
        cache_recovery = AutoRecoveryAction(
            name="emergency_cache_clear",
            trigger_condition=cache_clear_trigger,
            recovery_function=cache_clear_action,
            cooldown_seconds=300.0,
            max_executions=3
        )
        self.recovery_manager.register_recovery_action(cache_recovery)
    
    def _health_status_callback(self, metrics: Dict[str, HealthMetric]):
        """Callback for health status changes."""
        # Update system state for recovery manager
        state_updates = {
            f"{name}": metric.value for name, metric in metrics.items()
        }
        self.recovery_manager.update_system_state(state_updates)
        
        # Check and execute recovery actions
        self.recovery_manager.check_and_execute_recovery()
        
        # Update resilience level
        self._update_resilience_level(metrics)
    
    def _update_resilience_level(self, metrics: Dict[str, HealthMetric]):
        """Update overall resilience level based on health metrics."""
        critical_count = sum(1 for m in metrics.values() if m.status == ComponentStatus.CRITICAL)
        degraded_count = sum(1 for m in metrics.values() if m.status == ComponentStatus.DEGRADED)
        
        previous_level = self.resilience_level
        
        if critical_count > 2:
            self.resilience_level = ResilienceLevel.EMERGENCY
        elif critical_count > 0:
            self.resilience_level = ResilienceLevel.CRITICAL
        elif degraded_count > 1:
            self.resilience_level = ResilienceLevel.DEGRADED
        else:
            self.resilience_level = ResilienceLevel.NORMAL
        
        if self.resilience_level != previous_level:
            logger.warning(f"Resilience level changed: {previous_level.value} → {self.resilience_level.value}")
    
    def execute_with_resilience(self, operation_name: str, func: Callable, *args, **kwargs):
        """Execute function with full resilience protection."""
        if not self._initialized:
            self.initialize()
        
        # Get or create circuit breaker for this operation
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = AdvancedCircuitBreaker(operation_name)
        
        circuit_breaker = self.circuit_breakers[operation_name]
        
        # Execute with circuit breaker protection
        return circuit_breaker.call(func, *args, **kwargs)
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        health_status = self.health_monitor.get_overall_health()
        
        circuit_breaker_status = {
            name: breaker.get_metrics() 
            for name, breaker in self.circuit_breakers.items()
        }
        
        return {
            "resilience_level": self.resilience_level.value,
            "health": health_status,
            "circuit_breakers": circuit_breaker_status,
            "recovery_actions": [
                {
                    "name": action.name,
                    "execution_count": action.execution_count,
                    "last_executed": action.last_executed,
                    "max_executions": action.max_executions
                }
                for action in self.recovery_manager.recovery_actions
            ],
            "timestamp": time.time()
        }
    
    def shutdown(self):
        """Shutdown resilience framework."""
        self.health_monitor.stop_monitoring()
        logger.info("Resilience framework shutdown complete")


# Global resilience manager instance
resilience_manager = AutonomousResilienceManager()


def with_resilience(operation_name: str):
    """Decorator for adding resilience to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return resilience_manager.execute_with_resilience(operation_name, func, *args, **kwargs)
        return wrapper
    return decorator


def get_resilience_manager() -> AutonomousResilienceManager:
    """Get the global resilience manager instance."""
    return resilience_manager
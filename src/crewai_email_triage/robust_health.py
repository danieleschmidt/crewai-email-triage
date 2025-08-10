"""Health monitoring and metrics collection."""

import time
import psutil
import threading
import logging
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    status: HealthStatus
    message: str
    threshold_warning: float = 0.0
    threshold_critical: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class SystemHealth:
    """Overall system health information."""
    status: HealthStatus
    metrics: List[HealthMetric]
    overall_score: float  # 0-100
    response_time_ms: float
    timestamp: float = field(default_factory=time.time)

class HealthMonitor:
    """System health monitoring."""
    
    def __init__(self):
        self.metrics_history = []
        self.is_monitoring = False
        self._monitor_thread = None
        
    def check_system_health(self) -> SystemHealth:
        """Check comprehensive system health."""
        start_time = time.time()
        metrics = []
        
        # CPU Health
        cpu_metric = self._check_cpu_health()
        metrics.append(cpu_metric)
        
        # Memory Health
        memory_metric = self._check_memory_health()
        metrics.append(memory_metric)
        
        # Disk Health
        disk_metric = self._check_disk_health()
        metrics.append(disk_metric)
        
        # Process Health
        process_metric = self._check_process_health()
        metrics.append(process_metric)
        
        # Calculate overall health
        overall_status, overall_score = self._calculate_overall_health(metrics)
        
        response_time = (time.time() - start_time) * 1000
        
        return SystemHealth(
            status=overall_status,
            metrics=metrics,
            overall_score=overall_score,
            response_time_ms=response_time
        )
    
    def _check_cpu_health(self) -> HealthMetric:
        """Check CPU health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            elif cpu_percent > 70:
                status = HealthStatus.DEGRADED
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                status=status,
                message=message,
                threshold_warning=70.0,
                threshold_critical=90.0
            )
        except Exception as e:
            return HealthMetric(
                name="cpu_usage",
                value=-1,
                status=HealthStatus.UNHEALTHY,
                message=f"CPU check failed: {e}"
            )
    
    def _check_memory_health(self) -> HealthMetric:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_percent:.1f}%"
            elif memory_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthMetric(
                name="memory_usage",
                value=memory_percent,
                status=status,
                message=message,
                threshold_warning=80.0,
                threshold_critical=90.0
            )
        except Exception as e:
            return HealthMetric(
                name="memory_usage",
                value=-1,
                status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e}"
            )
    
    def _check_disk_health(self) -> HealthMetric:
        """Check disk health."""
        try:
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            if disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = f"Disk usage critical: {disk_percent:.1f}%"
            elif disk_percent > 85:
                status = HealthStatus.DEGRADED
                message = f"Disk usage high: {disk_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {disk_percent:.1f}%"
            
            return HealthMetric(
                name="disk_usage",
                value=disk_percent,
                status=status,
                message=message,
                threshold_warning=85.0,
                threshold_critical=95.0
            )
        except Exception as e:
            return HealthMetric(
                name="disk_usage",
                value=-1,
                status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {e}"
            )
    
    def _check_process_health(self) -> HealthMetric:
        """Check process health."""
        try:
            process_count = len(psutil.pids())
            
            if process_count > 1000:
                status = HealthStatus.DEGRADED
                message = f"High process count: {process_count}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Process count normal: {process_count}"
            
            return HealthMetric(
                name="process_count",
                value=process_count,
                status=status,
                message=message,
                threshold_warning=800.0,
                threshold_critical=1000.0
            )
        except Exception as e:
            return HealthMetric(
                name="process_count",
                value=-1,
                status=HealthStatus.UNHEALTHY,
                message=f"Process check failed: {e}"
            )
    
    def _calculate_overall_health(self, metrics: List[HealthMetric]) -> tuple[HealthStatus, float]:
        """Calculate overall health status and score."""
        if not metrics:
            return HealthStatus.UNHEALTHY, 0.0
        
        unhealthy_count = sum(1 for m in metrics if m.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for m in metrics if m.status == HealthStatus.DEGRADED)
        healthy_count = len(metrics) - unhealthy_count - degraded_count
        
        # Calculate score (0-100)
        score = (healthy_count * 100 + degraded_count * 60) / len(metrics)
        
        # Determine overall status
        if unhealthy_count > 0:
            status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        return status, score
    
    def start_continuous_monitoring(self, interval: float = 60.0):
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._continuous_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Health monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _continuous_monitor_loop(self, interval: float):
        """Continuous monitoring loop."""
        while self.is_monitoring:
            try:
                health = self.check_system_health()
                self.metrics_history.append(health)
                
                # Keep only last 100 entries
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                # Log critical issues
                if health.status == HealthStatus.UNHEALTHY:
                    logger.error(f"System health UNHEALTHY (score: {health.overall_score:.1f})")
                elif health.status == HealthStatus.DEGRADED:
                    logger.warning(f"System health DEGRADED (score: {health.overall_score:.1f})")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(interval)

# Global health monitor
_health_monitor = HealthMonitor()

def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    return _health_monitor

def quick_health_check() -> Dict[str, Any]:
    """Quick health check returning simplified results."""
    health = _health_monitor.check_system_health()
    
    return {
        "status": health.status.value,
        "score": health.overall_score,
        "response_time_ms": health.response_time_ms,
        "timestamp": health.timestamp,
        "issues": [
            f"{m.name}: {m.message}" for m in health.metrics 
            if m.status != HealthStatus.HEALTHY
        ]
    }

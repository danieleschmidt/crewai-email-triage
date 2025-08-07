"""Health checking and system monitoring for CrewAI Email Triage."""

from __future__ import annotations

import threading
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .metrics_export import get_metrics_collector
from .rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    name: str
    status: HealthStatus
    message: str
    response_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp,
            "details": self.details
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    status: HealthStatus
    checks: List[HealthCheckResult]
    response_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp,
            "checks": [check.to_dict() for check in self.checks]
        }


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._register_default_checks()
        self._metrics_collector = get_metrics_collector()

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self.register_check("memory", self._check_memory)
        self.register_check("cpu", self._check_cpu)
        self.register_check("disk", self._check_disk)
        self.register_check("agents", self._check_agents)
        self.register_check("metrics", self._check_metrics)
        self.register_check("rate_limiter", self._check_rate_limiter)

    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Register a custom health check."""
        self._checks[name] = check_func
        logger.debug("Registered health check: %s", name)

    def _check_memory(self) -> HealthCheckResult:
        """Check system memory usage."""
        start_time = time.perf_counter()
        try:
            if not PSUTIL_AVAILABLE:
                status = HealthStatus.UNKNOWN
                message = "Memory check unavailable (psutil not installed)"
                details = {"error": "psutil not available"}
            else:
                memory = psutil.virtual_memory()
                used_percent = memory.percent

                if used_percent < 70:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage is normal ({used_percent:.1f}%)"
                elif used_percent < 85:
                    status = HealthStatus.DEGRADED
                    message = f"Memory usage is elevated ({used_percent:.1f}%)"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"Memory usage is critical ({used_percent:.1f}%)"

                details = {
                    "used_percent": used_percent,
                    "total_gb": memory.total / (1024**3),
                    "available_gb": memory.available / (1024**3)
                }

        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check memory: {e}"
            details = {"error": str(e)}

        response_time = (time.perf_counter() - start_time) * 1000
        return HealthCheckResult("memory", status, message, response_time, details=details)

    def _check_cpu(self) -> HealthCheckResult:
        """Check CPU usage."""
        start_time = time.perf_counter()
        try:
            if not PSUTIL_AVAILABLE:
                status = HealthStatus.UNKNOWN
                message = "CPU check unavailable (psutil not installed)"
                details = {"error": "psutil not available"}
            else:
                # Get CPU usage over a short interval
                cpu_percent = psutil.cpu_percent(interval=0.1)

                if cpu_percent < 70:
                    status = HealthStatus.HEALTHY
                    message = f"CPU usage is normal ({cpu_percent:.1f}%)"
                elif cpu_percent < 85:
                    status = HealthStatus.DEGRADED
                    message = f"CPU usage is elevated ({cpu_percent:.1f}%)"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"CPU usage is critical ({cpu_percent:.1f}%)"

                details = {
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count(),
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }

        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check CPU: {e}"
            details = {"error": str(e)}

        response_time = (time.perf_counter() - start_time) * 1000
        return HealthCheckResult("cpu", status, message, response_time, details=details)

    def _check_disk(self) -> HealthCheckResult:
        """Check disk usage."""
        start_time = time.perf_counter()
        try:
            if not PSUTIL_AVAILABLE:
                status = HealthStatus.UNKNOWN
                message = "Disk check unavailable (psutil not installed)"
                details = {"error": "psutil not available"}
            else:
                disk = psutil.disk_usage('/')
                used_percent = (disk.used / disk.total) * 100

                if used_percent < 80:
                    status = HealthStatus.HEALTHY
                    message = f"Disk usage is normal ({used_percent:.1f}%)"
                elif used_percent < 90:
                    status = HealthStatus.DEGRADED
                    message = f"Disk usage is elevated ({used_percent:.1f}%)"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"Disk usage is critical ({used_percent:.1f}%)"

                details = {
                    "used_percent": used_percent,
                    "total_gb": disk.total / (1024**3),
                    "free_gb": disk.free / (1024**3)
                }

        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check disk: {e}"
            details = {"error": str(e)}

        response_time = (time.perf_counter() - start_time) * 1000
        return HealthCheckResult("disk", status, message, response_time, details=details)

    def _check_agents(self) -> HealthCheckResult:
        """Check agent system health."""
        start_time = time.perf_counter()
        try:
            from .pipeline import triage_email_enhanced

            # Test with a simple message
            test_result = triage_email_enhanced("Health check test message")

            if test_result.success and not test_result.errors:
                status = HealthStatus.HEALTHY
                message = "All agents functioning normally"
            elif test_result.partial_success:
                status = HealthStatus.DEGRADED
                message = f"Some agent issues detected: {', '.join(test_result.errors[:3])}"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Agent system failing: {', '.join(test_result.errors[:3])}"

            details = {
                "success": test_result.success,
                "partial_success": test_result.partial_success,
                "errors": test_result.errors,
                "warnings": test_result.warnings,
                "processing_time_ms": test_result.processing_time_ms
            }

        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check agents: {e}"
            details = {"error": str(e)}

        response_time = (time.perf_counter() - start_time) * 1000
        return HealthCheckResult("agents", status, message, response_time, details=details)

    def _check_metrics(self) -> HealthCheckResult:
        """Check metrics collection system."""
        start_time = time.perf_counter()
        try:
            # Test metrics collection
            self._metrics_collector.increment_counter("health_check_test")
            counter_value = self._metrics_collector.get_counter("health_check_test")

            if counter_value > 0:
                status = HealthStatus.HEALTHY
                message = "Metrics system functioning normally"
            else:
                status = HealthStatus.DEGRADED
                message = "Metrics collection may have issues"

            details = {
                "test_counter_value": counter_value,
                "metrics_enabled": True
            }

        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check metrics: {e}"
            details = {"error": str(e)}

        response_time = (time.perf_counter() - start_time) * 1000
        return HealthCheckResult("metrics", status, message, response_time, details=details)

    def _check_rate_limiter(self) -> HealthCheckResult:
        """Check rate limiter health."""
        start_time = time.perf_counter()
        try:
            rate_limiter = get_rate_limiter()
            limiter_status = rate_limiter.get_status()

            if limiter_status["utilization"] < 0.8:
                status = HealthStatus.HEALTHY
                message = f"Rate limiter healthy (utilization: {limiter_status['utilization']:.1%})"
            elif limiter_status["utilization"] < 0.95:
                status = HealthStatus.DEGRADED
                message = f"Rate limiter under pressure (utilization: {limiter_status['utilization']:.1%})"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Rate limiter saturated (utilization: {limiter_status['utilization']:.1%})"

            details = {
                "enabled": limiter_status["enabled"],
                "utilization": limiter_status["utilization"],
                "tokens_available": limiter_status["tokens_available"],
                "requests_per_minute": limiter_status.get("requests_per_minute", 0)
            }

        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Failed to check rate limiter: {e}"
            details = {"error": str(e)}

        response_time = (time.perf_counter() - start_time) * 1000
        return HealthCheckResult("rate_limiter", status, message, response_time, details=details)

    def check_health(self, checks: Optional[List[str]] = None) -> SystemHealth:
        """Perform health checks and return overall system health."""
        start_time = time.perf_counter()

        if checks is None:
            checks_to_run = list(self._checks.keys())
        else:
            checks_to_run = [check for check in checks if check in self._checks]

        results = []
        for check_name in checks_to_run:
            try:
                result = self._checks[check_name]()
                results.append(result)
                logger.debug("Health check %s: %s", check_name, result.status.value)
            except Exception as e:
                logger.error("Health check %s failed: %s", check_name, e, exc_info=True)
                results.append(HealthCheckResult(
                    check_name,
                    HealthStatus.UNKNOWN,
                    f"Check failed: {e}",
                    details={"error": str(e)}
                ))

        # Determine overall health status
        if not results:
            overall_status = HealthStatus.UNKNOWN
        elif all(r.status == HealthStatus.HEALTHY for r in results):
            overall_status = HealthStatus.HEALTHY
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        response_time = (time.perf_counter() - start_time) * 1000

        # Update metrics
        self._metrics_collector.set_gauge("system_health_status",
            1 if overall_status == HealthStatus.HEALTHY else 0)
        self._metrics_collector.record_histogram("health_check_duration_ms", response_time)

        return SystemHealth(overall_status, results, response_time)


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


class HealthMonitor:
    """Continuous health monitoring service."""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._health_checker = get_health_checker()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._metrics_collector = get_metrics_collector()

    def start(self) -> None:
        """Start continuous health monitoring."""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitor started with %ds interval", self.check_interval)

    def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Health monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                health = self._health_checker.check_health()

                # Log health status changes
                if health.status != HealthStatus.HEALTHY:
                    logger.warning("System health: %s", health.status.value)
                    for check in health.checks:
                        if check.status != HealthStatus.HEALTHY:
                            logger.warning("Health check %s: %s - %s",
                                         check.name, check.status.value, check.message)

                # Update metrics
                self._metrics_collector.set_gauge("health_monitor_last_check", time.time())

            except Exception as e:
                logger.error("Health monitoring error: %s", e, exc_info=True)

            # Wait for next check
            time.sleep(self.check_interval)

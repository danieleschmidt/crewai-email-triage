"""Comprehensive error handling with circuit breakers and retries."""

import functools
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("Circuit breaker CLOSED - service recovered")

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error(f"Circuit breaker OPEN - {self.failure_count} consecutive failures")

class RobustErrorHandler:
    """Comprehensive error handling system."""

    def __init__(self):
        self.circuit_breaker = CircuitBreaker()
        self.error_metrics = {
            "total_errors": 0,
            "error_by_type": {},
            "error_by_severity": {severity.value: 0 for severity in ErrorSeverity}
        }

    def handle_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: str = "unknown") -> Dict[str, Any]:
        """Handle error with comprehensive logging and metrics."""
        error_type = type(error).__name__

        # Update metrics
        self.error_metrics["total_errors"] += 1
        self.error_metrics["error_by_type"][error_type] =             self.error_metrics["error_by_type"].get(error_type, 0) + 1
        self.error_metrics["error_by_severity"][severity.value] += 1

        # Log error with context
        log_msg = f"Error in {context}: {error_type} - {str(error)}"

        if severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif severity == ErrorSeverity.HIGH:
            logger.error(log_msg)
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        return {
            "error_type": error_type,
            "error_message": str(error),
            "severity": severity.value,
            "context": context,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "handled": True
        }

    def get_error_metrics(self) -> Dict[str, Any]:
        """Get current error metrics."""
        return self.error_metrics.copy()

def with_error_handling(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       context: str = "operation"):
    """Decorator for adding error handling to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = RobustErrorHandler()
                error_info = handler.handle_error(e, severity, context)

                # Return error info instead of crashing
                return {
                    "success": False,
                    "error": error_info,
                    "result": None
                }
        return wrapper
    return decorator

# Global error handler instance
_error_handler = RobustErrorHandler()

def get_error_handler() -> RobustErrorHandler:
    """Get the global error handler instance."""
    return _error_handler

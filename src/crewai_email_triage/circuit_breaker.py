"""Circuit breaker pattern implementation for preventing cascading failures."""

import time
import threading
import logging
from typing import Any, Callable, Dict, Tuple, Type, Optional
from enum import Enum
import os

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        monitored_exceptions: Tuple[Type[Exception], ...] = (
            ConnectionError,
            TimeoutError,
            OSError,
        )
    ):
        """Initialize circuit breaker configuration.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds to wait before trying half-open
            success_threshold: Number of successes needed to close circuit from half-open
            monitored_exceptions: Exception types that trigger circuit breaker
        """
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if recovery_timeout < 0:
            raise ValueError("recovery_timeout must be non-negative")
        if success_threshold <= 0:
            raise ValueError("success_threshold must be positive")
            
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.monitored_exceptions = monitored_exceptions
    
    @classmethod
    def from_env(cls) -> "CircuitBreakerConfig":
        """Create circuit breaker config from environment variables."""
        return cls(
            failure_threshold=int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")),
            recovery_timeout=float(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60.0")),
            success_threshold=int(os.getenv("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", "3"))
        )


class CircuitBreaker:
    """Circuit breaker implementation to prevent cascading failures."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Name identifier for this circuit breaker
            config: Configuration for circuit breaker behavior
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # Thread-safe state management
        self._lock = threading.RLock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._total_calls = 0
        
        logger.info(
            "Circuit breaker '%s' initialized with failure_threshold=%d, recovery_timeout=%.1fs",
            self.name, self.config.failure_threshold, self.config.recovery_timeout
        )
    
    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._state.value
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._failure_count
    
    @property
    def success_count(self) -> int:
        """Get current success count."""
        with self._lock:
            return self._success_count
    
    def call(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation through circuit breaker.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Result of operation
            
        Raises:
            CircuitBreakerError: When circuit is open
            Any exception raised by operation
        """
        with self._lock:
            self._total_calls += 1
            
            # Check if we should transition to half-open
            if self._state == CircuitState.OPEN and self._should_attempt_reset():
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info("Circuit breaker '%s' transitioning to half-open state", self.name)
            
            # Fast-fail if circuit is open
            if self._state == CircuitState.OPEN:
                logger.warning(
                    "Circuit breaker '%s' is open, fast-failing operation",
                    self.name
                )
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Last failure: {self._last_failure_time}"
                )
        
        # Execute operation outside the lock to avoid deadlocks
        try:
            result = operation(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.monitored_exceptions as e:
            self._on_failure(e)
            raise
            
        except Exception as e:
            # Non-monitored exceptions pass through without affecting circuit state
            logger.debug(
                "Circuit breaker '%s' ignoring non-monitored exception: %s",
                self.name, type(e).__name__
            )
            raise
    
    def _on_success(self):
        """Handle successful operation."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    "Circuit breaker '%s' half-open success %d/%d",
                    self.name, self._success_count, self.config.success_threshold
                )
                
                if self._success_count >= self.config.success_threshold:
                    self._reset_to_closed()
                    logger.info(
                        "Circuit breaker '%s' closing after %d successful operations",
                        self.name, self._success_count
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success in closed state
                self._failure_count = 0
    
    def _on_failure(self, exception: Exception):
        """Handle failed operation.
        
        Args:
            exception: The exception that caused the failure
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            logger.warning(
                "Circuit breaker '%s' recorded failure %d/%d: %s",
                self.name, self._failure_count, self.config.failure_threshold, str(exception)
            )
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open state immediately opens circuit
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker '%s' opening due to failure in half-open state",
                    self.name
                )
            elif self._state == CircuitState.CLOSED and self._failure_count >= self.config.failure_threshold:
                # Threshold reached, open circuit
                self._state = CircuitState.OPEN
                logger.error(
                    "Circuit breaker '%s' opening after %d failures",
                    self.name, self._failure_count
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.config.recovery_timeout
    
    def _reset_to_closed(self):
        """Reset circuit to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._reset_to_closed()
            logger.info("Circuit breaker '%s' manually reset to closed state", self.name)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics.
        
        Returns:
            Dictionary containing current metrics
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": self._total_calls,
                "last_failure_time": self._last_failure_time,
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create a circuit breaker.
        
        Args:
            name: Name identifier for the circuit breaker
            config: Configuration for new circuit breakers
            
        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers.
        
        Returns:
            Dictionary mapping breaker names to their metrics
        """
        with self._lock:
            return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get a circuit breaker from the global registry.
    
    Args:
        name: Name identifier for the circuit breaker
        config: Configuration for new circuit breakers
        
    Returns:
        CircuitBreaker instance
    """
    return _registry.get_breaker(name, config)


def reset_all_circuit_breakers():
    """Reset all circuit breakers in the global registry."""
    _registry.reset_all()


def get_all_circuit_breaker_metrics() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all circuit breakers in the global registry.
    
    Returns:
        Dictionary mapping breaker names to their metrics
    """
    return _registry.get_all_metrics()
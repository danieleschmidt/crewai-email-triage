"""Retry utilities with exponential backoff and circuit breaker for network operations."""

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

from .circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerError,
    get_circuit_breaker,
)

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (
            ConnectionError,
            TimeoutError,
            OSError,
        ),
        enable_circuit_breaker: bool = True,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of attempts (including initial attempt)
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay in seconds
            exponential_factor: Factor for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that should trigger retries
            enable_circuit_breaker: Whether to enable circuit breaker pattern
            circuit_breaker_config: Configuration for circuit breaker behavior
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if max_delay < base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if exponential_factor <= 1:
            raise ValueError("exponential_factor must be > 1")

        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_factor = exponential_factor
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()

    @classmethod
    def from_env(cls) -> "RetryConfig":
        """Create retry config from environment variables."""
        from .env_config import get_retry_config

        env_config = get_retry_config()
        return cls(
            max_attempts=env_config.max_attempts,
            base_delay=env_config.base_delay,
            max_delay=env_config.max_delay,
            exponential_factor=env_config.exponential_factor,
            jitter=env_config.jitter,
            enable_circuit_breaker=True,  # Default enabled
            circuit_breaker_config=CircuitBreakerConfig.from_env()
        )


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for a given retry attempt.
    
    Args:
        attempt: Attempt number (1-based)
        config: Retry configuration
        
    Returns:
        Delay in seconds
    """
    if attempt <= 0:
        return 0.0

    # Exponential backoff: base_delay * (exponential_factor ^ (attempt-1))
    delay = config.base_delay * (config.exponential_factor ** (attempt - 1))

    # Cap at max_delay
    delay = min(delay, config.max_delay)

    # Add jitter if enabled
    if config.jitter:
        jitter_factor = random.uniform(0.5, 1.5)
        delay *= jitter_factor

    return delay


def retry_with_backoff(config: Optional[RetryConfig] = None, circuit_breaker_name: Optional[str] = None):
    """Decorator to add retry logic with exponential backoff and circuit breaker to functions.
    
    Args:
        config: Retry configuration. If None, uses default config.
        circuit_breaker_name: Name for circuit breaker. If None, uses function name.
        
    Returns:
        Decorated function with retry logic and circuit breaker
    """
    if config is None:
        config = RetryConfig.from_env()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            breaker_name = circuit_breaker_name or f"{func.__module__}.{func.__name__}"

            if config.enable_circuit_breaker:
                circuit_breaker = get_circuit_breaker(breaker_name, config.circuit_breaker_config)

                # Use circuit breaker to wrap the entire retry operation
                def retry_operation():
                    return _execute_with_retry(func, config, *args, **kwargs)

                try:
                    return circuit_breaker.call(retry_operation)
                except CircuitBreakerError:
                    logger.error("Circuit breaker is open, skipping retry logic", extra={
                        'circuit_breaker': breaker_name,
                        'function': func.__name__,
                        'operation': 'retry_with_circuit_breaker',
                        'error_type': 'circuit_breaker_open'
                    })
                    raise
            else:
                # Original retry logic without circuit breaker
                return _execute_with_retry(func, config, *args, **kwargs)

        return wrapper
    return decorator


def _execute_with_retry(func: Callable, config: RetryConfig, *args, **kwargs) -> Any:
    """Execute function with retry logic.
    
    Args:
        func: Function to execute
        config: Retry configuration
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function
        
    Returns:
        Result of function execution
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(1, config.max_attempts + 1):
        try:
            result = func(*args, **kwargs)
            if attempt > 1:
                logger.info("Function succeeded after retry", extra={
                    'function': func.__name__,
                    'attempt': attempt,
                    'max_attempts': config.max_attempts,
                    'operation': 'retry_execute',
                    'success': True
                })
            return result

        except config.retryable_exceptions as e:
            last_exception = e

            if attempt == config.max_attempts:
                logger.error("Function failed after all retry attempts", extra={
                    'function': func.__name__,
                    'max_attempts': config.max_attempts,
                    'final_error': str(e),
                    'error_type': type(e).__name__,
                    'operation': 'retry_execute',
                    'success': False
                })
                break

            delay = calculate_delay(attempt, config)
            logger.warning("Function failed, retrying with backoff", extra={
                'function': func.__name__,
                'attempt': attempt,
                'max_attempts': config.max_attempts,
                'error': str(e),
                'error_type': type(e).__name__,
                'retry_delay': delay,
                'operation': 'retry_execute'
            })

            time.sleep(delay)

        except Exception as e:
            # Non-retryable exception, re-raise immediately
            logger.error("Function failed with non-retryable exception", extra={
                'function': func.__name__,
                'error': str(e),
                'error_type': type(e).__name__,
                'operation': 'retry_execute',
                'retryable': False
            })
            raise

    # If we get here, all retries were exhausted
    raise last_exception


def retry_operation(
    operation: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    circuit_breaker_name: Optional[str] = None,
    **kwargs
) -> Any:
    """Execute an operation with retry logic and circuit breaker.
    
    Args:
        operation: Function to execute
        *args: Positional arguments for the operation
        config: Retry configuration
        circuit_breaker_name: Name for circuit breaker. If None, uses operation name.
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of the operation
        
    Raises:
        The last exception if all retries fail
        CircuitBreakerError: If circuit breaker is open
    """
    decorated_operation = retry_with_backoff(config, circuit_breaker_name)(operation)
    return decorated_operation(*args, **kwargs)

"""Retry utilities with exponential backoff for network operations."""

import time
import random
import logging
from functools import wraps
from typing import Any, Callable, Type, Tuple, Union, Optional
import os

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
        )
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of attempts (including initial attempt)
            base_delay: Base delay in seconds for first retry
            max_delay: Maximum delay in seconds
            exponential_factor: Factor for exponential backoff
            jitter: Whether to add random jitter to delays
            retryable_exceptions: Tuple of exception types that should trigger retries
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
            jitter=env_config.jitter
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


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic with exponential backoff to functions.
    
    Args:
        config: Retry configuration. If None, uses default config.
        
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig.from_env()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(
                            "Function %s succeeded on attempt %d/%d",
                            func.__name__, attempt, config.max_attempts
                        )
                    return result
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            "Function %s failed after %d attempts. Final error: %s",
                            func.__name__, config.max_attempts, str(e)
                        )
                        break
                    
                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        "Function %s failed on attempt %d/%d (error: %s). Retrying in %.2f seconds...",
                        func.__name__, attempt, config.max_attempts, str(e), delay
                    )
                    
                    time.sleep(delay)
                    
                except Exception as e:
                    # Non-retryable exception, re-raise immediately
                    logger.error(
                        "Function %s failed with non-retryable exception: %s",
                        func.__name__, str(e)
                    )
                    raise
            
            # If we get here, all retries were exhausted
            raise last_exception
        
        return wrapper
    return decorator


def retry_operation(
    operation: Callable,
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs
) -> Any:
    """Execute an operation with retry logic.
    
    Args:
        operation: Function to execute
        *args: Positional arguments for the operation
        config: Retry configuration
        **kwargs: Keyword arguments for the operation
        
    Returns:
        Result of the operation
        
    Raises:
        The last exception if all retries fail
    """
    decorated_operation = retry_with_backoff(config)(operation)
    return decorated_operation(*args, **kwargs)
"""Rate limiting and backpressure mechanisms for email processing pipeline.

Implements configurable rate limiting using token bucket algorithm to prevent
service degradation under high load and protect against API quota exhaustion.
"""

import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional

from .env_config import get_rate_limit_config


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting behavior."""
    
    requests_per_second: float = 10.0
    """Maximum requests allowed per second."""
    
    burst_size: int = 20
    """Maximum burst capacity (token bucket size)."""
    
    enabled: bool = True
    """Whether rate limiting is enabled."""
    
    backpressure_threshold: float = 0.8
    """Threshold (0-1) for activating backpressure when token bucket is low."""
    
    backpressure_delay: float = 0.1
    """Additional delay when backpressure is active."""


class RateLimiter:
    """Thread-safe token bucket rate limiter with backpressure support.
    
    Uses token bucket algorithm to provide smooth rate limiting with burst capacity.
    Implements backpressure mechanism to provide early warnings when approaching limits.
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter with configuration.
        
        Parameters
        ----------
        config : RateLimitConfig, optional
            Rate limiting configuration. If None, uses environment defaults.
        """
        self._config = config or self._load_env_config()
        self._lock = threading.RLock()
        
        # Token bucket state
        self._tokens = float(self._config.burst_size)
        self._last_refill = time.time()
        
        # Backpressure tracking
        self._backpressure_active = False
        
    def _load_env_config(self) -> RateLimitConfig:
        """Load rate limiting configuration from environment variables."""
        env_config = get_rate_limit_config()
        return RateLimitConfig(
            requests_per_second=env_config.requests_per_second,
            burst_size=env_config.burst_size,
            enabled=env_config.enabled,
            backpressure_threshold=env_config.backpressure_threshold,
            backpressure_delay=env_config.backpressure_delay,
        )
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        current_time = time.time()
        elapsed = current_time - self._last_refill
        
        # Add tokens based on rate and elapsed time
        tokens_to_add = elapsed * self._config.requests_per_second
        self._tokens = min(self._config.burst_size, self._tokens + tokens_to_add)
        self._last_refill = current_time
        
        # Update backpressure state
        token_ratio = self._tokens / self._config.burst_size
        self._backpressure_active = token_ratio < self._config.backpressure_threshold
    
    def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens from the bucket with optional backpressure delay.
        
        Parameters
        ----------
        tokens : float, optional
            Number of tokens to acquire. Default is 1.0.
            
        Returns
        -------
        float
            Delay time that was applied (0 if no delay needed).
        """
        if not self._config.enabled:
            return 0.0
            
        with self._lock:
            self._refill_tokens()
            
            # Check if we have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                
                # Apply backpressure delay if needed
                if self._backpressure_active:
                    time.sleep(self._config.backpressure_delay)
                    return self._config.backpressure_delay
                    
                return 0.0
            
            # Not enough tokens - calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / self._config.requests_per_second
            
            # Wait and then acquire
            time.sleep(wait_time)
            self._tokens = 0.0  # All tokens consumed
            
            return wait_time
    
    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without waiting.
        
        Parameters
        ----------
        tokens : float, optional
            Number of tokens to acquire. Default is 1.0.
            
        Returns
        -------
        bool
            True if tokens were acquired, False if not enough available.
        """
        if not self._config.enabled:
            return True
            
        with self._lock:
            self._refill_tokens()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
                
            return False
    
    def get_status(self) -> Dict[str, float]:
        """Get current rate limiter status.
        
        Returns
        -------
        Dict[str, float]
            Status information including available tokens, utilization, etc.
        """
        with self._lock:
            self._refill_tokens()
            
            return {
                "enabled": self._config.enabled,
                "tokens_available": self._tokens,
                "max_tokens": self._config.burst_size,
                "utilization": 1.0 - (self._tokens / self._config.burst_size),
                "backpressure_active": self._backpressure_active,
                "requests_per_second": self._config.requests_per_second,
            }
    
    @contextmanager
    def rate_limited_operation(self, tokens: float = 1.0):
        """Context manager for rate-limited operations.
        
        Parameters
        ----------
        tokens : float, optional
            Number of tokens to acquire for this operation. Default is 1.0.
            
        Yields
        ------
        float
            Delay time that was applied.
        """
        delay = self.acquire(tokens)
        try:
            yield delay
        finally:
            # Could implement token return here if operation fails
            pass


class BatchRateLimiter:
    """Rate limiter optimized for batch processing with adaptive throttling."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize batch rate limiter.
        
        Parameters
        ----------
        config : RateLimitConfig, optional
            Rate limiting configuration. If None, uses environment defaults.
        """
        self._rate_limiter = RateLimiter(config)
        self._config = self._rate_limiter._config
        
    def process_batch_with_rate_limiting(self, batch_size: int, processing_func, items):
        """Process a batch of items with rate limiting and backpressure.
        
        Parameters
        ----------
        batch_size : int
            Number of items in the batch.
        processing_func : callable
            Function to process each item.
        items : iterable
            Items to process.
            
        Yields
        ------
        Any
            Results from processing_func.
        """
        if not self._config.enabled:
            # No rate limiting - process normally
            for item in items:
                yield processing_func(item)
            return
            
        # Calculate adaptive batch delay
        status = self._rate_limiter.get_status()
        
        # If utilization is high, add extra delay between batches
        if status["utilization"] > 0.9:
            batch_delay = 1.0 / self._config.requests_per_second
            time.sleep(batch_delay)
        
        # Process items with individual rate limiting
        for item in items:
            with self._rate_limiter.rate_limited_operation() as delay:
                yield processing_func(item)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = RateLimiter()
    
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (mainly for testing)."""
    global _rate_limiter
    with _rate_limiter_lock:
        _rate_limiter = None
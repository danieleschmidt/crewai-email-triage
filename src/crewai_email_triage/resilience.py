"""Advanced resilience and fault tolerance mechanisms."""

from __future__ import annotations

import asyncio
import functools
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from enum import Enum

from .logging_utils import get_logger
from .metrics_export import get_metrics_collector

logger = get_logger(__name__)
_metrics_collector = get_metrics_collector()

T = TypeVar('T')


class FailureMode(Enum):
    """Types of failure modes the system can handle."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    AUTHENTICATION_FAILURE = "authentication_failure"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ResilienceMetrics:
    """Metrics for resilience operations."""
    
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    retries: int = 0
    circuit_breaker_trips: int = 0
    fallback_invocations: int = 0
    average_response_time_ms: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "retries": self.retries,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "fallback_invocations": self.fallback_invocations,
            "success_rate": self.success_rate(),
            "average_response_time_ms": self.average_response_time_ms,
        }


class BulkheadIsolation:
    """Bulkhead pattern implementation for resource isolation."""
    
    def __init__(self, max_concurrent: int = 10, timeout: float = 30.0):
        """Initialize bulkhead isolation.
        
        Args:
            max_concurrent: Maximum concurrent operations allowed
            timeout: Timeout for operations in seconds
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.active_operations = 0
        
        logger.info(f"BulkheadIsolation initialized with max_concurrent={max_concurrent}")
    
    async def execute(self, operation: Callable[[], T]) -> T:
        """Execute operation with bulkhead isolation."""
        
        async with self.semaphore:
            self.active_operations += 1
            _metrics_collector.set_gauge("bulkhead_active_operations", self.active_operations)
            
            try:
                # Run the operation in thread pool with timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, operation),
                    timeout=self.timeout
                )
                
                _metrics_collector.increment_counter("bulkhead_successful_operations")
                return result
                
            except asyncio.TimeoutError:
                _metrics_collector.increment_counter("bulkhead_timeout_errors")
                raise TimeoutError(f"Operation timed out after {self.timeout}s")
            except Exception as e:
                _metrics_collector.increment_counter("bulkhead_operation_errors")
                raise
            finally:
                self.active_operations -= 1
                _metrics_collector.set_gauge("bulkhead_active_operations", self.active_operations)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bulkhead status."""
        available_permits = self.semaphore._value
        utilization = (self.max_concurrent - available_permits) / self.max_concurrent
        
        return {
            "max_concurrent": self.max_concurrent,
            "active_operations": self.active_operations,
            "available_permits": available_permits,
            "utilization": utilization,
            "timeout": self.timeout,
        }


class GracefulDegradation:
    """Implements graceful degradation patterns."""
    
    def __init__(self, fallback_strategies: Dict[str, Callable] = None):
        """Initialize graceful degradation.
        
        Args:
            fallback_strategies: Dict of fallback functions for different operations
        """
        self.fallback_strategies = fallback_strategies or {}
        self.degradation_level = 0  # 0 = normal, higher = more degraded
        self.max_degradation_level = 5
        
        logger.info("GracefulDegradation initialized")
    
    def set_degradation_level(self, level: int):
        """Set system degradation level."""
        self.degradation_level = max(0, min(level, self.max_degradation_level))
        _metrics_collector.set_gauge("system_degradation_level", self.degradation_level)
        
        logger.info(f"System degradation level set to {self.degradation_level}")
    
    def execute_with_fallback(
        self,
        primary_operation: Callable[[], T],
        operation_name: str,
        fallback_operation: Optional[Callable[[], T]] = None
    ) -> T:
        """Execute operation with fallback strategy."""
        
        start_time = time.perf_counter()
        
        try:
            # Skip expensive operations at high degradation levels
            if self.degradation_level >= 4 and operation_name in ['ai_analysis', 'complex_processing']:
                if fallback_operation:
                    logger.info(f"Using fallback for {operation_name} due to high degradation level")
                    _metrics_collector.increment_counter("graceful_degradation_fallbacks")
                    return fallback_operation()
                else:
                    raise Exception("Operation skipped due to degradation")
            
            # Execute primary operation
            result = primary_operation()
            
            processing_time = (time.perf_counter() - start_time) * 1000
            _metrics_collector.record_histogram(f"operation_{operation_name}_time_ms", processing_time)
            
            return result
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            _metrics_collector.increment_counter(f"operation_{operation_name}_failures")
            
            logger.warning(f"Primary operation {operation_name} failed: {e}")
            
            # Try registered fallback strategy
            fallback = fallback_operation or self.fallback_strategies.get(operation_name)
            if fallback:
                try:
                    logger.info(f"Executing fallback for {operation_name}")
                    _metrics_collector.increment_counter("graceful_degradation_fallbacks")
                    return fallback()
                except Exception as fallback_error:
                    logger.error(f"Fallback for {operation_name} also failed: {fallback_error}")
                    _metrics_collector.increment_counter("graceful_degradation_fallback_failures")
            
            # Re-raise original exception if no fallback available
            raise
    
    def register_fallback(self, operation_name: str, fallback_func: Callable):
        """Register a fallback function for an operation."""
        self.fallback_strategies[operation_name] = fallback_func
        logger.info(f"Registered fallback strategy for {operation_name}")


class AdaptiveRetry:
    """Adaptive retry mechanism with intelligent backoff."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        """Initialize adaptive retry.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Whether to add jitter to delays
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.failure_counts = {}  # Track failures per operation type
        
        logger.info(f"AdaptiveRetry initialized with max_attempts={max_attempts}")
    
    def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "unknown",
        retryable_exceptions: tuple = (Exception,)
    ) -> T:
        """Execute operation with adaptive retry."""
        
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                start_time = time.perf_counter()
                result = operation()
                
                processing_time = (time.perf_counter() - start_time) * 1000
                _metrics_collector.record_histogram(f"retry_operation_{operation_name}_time_ms", processing_time)
                
                # Reset failure count on success
                if operation_name in self.failure_counts:
                    del self.failure_counts[operation_name]
                
                if attempt > 1:
                    logger.info(f"Operation {operation_name} succeeded on attempt {attempt}")
                    _metrics_collector.increment_counter(f"retry_operation_{operation_name}_success_after_retry")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not isinstance(e, retryable_exceptions):
                    logger.error(f"Non-retryable exception for {operation_name}: {e}")
                    _metrics_collector.increment_counter(f"retry_operation_{operation_name}_non_retryable")
                    raise
                
                # Track failure
                self.failure_counts[operation_name] = self.failure_counts.get(operation_name, 0) + 1
                
                if attempt == self.max_attempts:
                    logger.error(f"Operation {operation_name} failed after {attempt} attempts: {e}")
                    _metrics_collector.increment_counter(f"retry_operation_{operation_name}_max_attempts_exceeded")
                    raise
                
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt, operation_name)
                
                logger.warning(f"Operation {operation_name} failed on attempt {attempt}, retrying in {delay:.2f}s: {e}")
                _metrics_collector.increment_counter(f"retry_operation_{operation_name}_attempt_{attempt}")
                
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception
    
    def _calculate_delay(self, attempt: int, operation_name: str) -> float:
        """Calculate adaptive delay based on attempt and failure history."""
        
        # Base exponential backoff
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        
        # Adaptive adjustment based on failure history
        failure_count = self.failure_counts.get(operation_name, 0)
        if failure_count > 5:
            delay *= 1.5  # Increase delay for frequently failing operations
        
        # Cap at maximum delay
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)  # Ensure minimum delay


class HealthCheck:
    """Advanced health check system for components."""
    
    def __init__(self):
        """Initialize health check system."""
        self.component_health = {}
        self.health_thresholds = {
            'response_time_ms': 5000,
            'error_rate': 0.1,
            'memory_usage_mb': 1000,
            'cpu_usage_percent': 80,
        }
        
        logger.info("HealthCheck system initialized")
    
    def check_component_health(self, component_name: str) -> Dict[str, Any]:
        """Check health of a specific component."""
        
        start_time = time.perf_counter()
        
        try:
            # Component-specific health checks
            if component_name == "email_processor":
                health = self._check_email_processor_health()
            elif component_name == "ai_analyzer":
                health = self._check_ai_analyzer_health()
            elif component_name == "security_scanner":
                health = self._check_security_scanner_health()
            else:
                health = self._check_generic_component_health(component_name)
            
            check_time = (time.perf_counter() - start_time) * 1000
            health['check_time_ms'] = check_time
            
            # Update component health cache
            self.component_health[component_name] = health
            
            _metrics_collector.record_histogram(f"health_check_{component_name}_time_ms", check_time)
            
            return health
            
        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            _metrics_collector.increment_counter(f"health_check_{component_name}_failures")
            
            return {
                'status': 'unhealthy',
                'error': str(e),
                'check_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    def _check_email_processor_health(self) -> Dict[str, Any]:
        """Check email processor component health."""
        
        # Check recent processing metrics
        recent_emails = _metrics_collector.get_counter("emails_processed")
        recent_errors = _metrics_collector.get_counter("email_processing_errors")
        
        error_rate = recent_errors / max(1, recent_emails)
        avg_response_time = _metrics_collector.get_gauge("avg_processing_time_ms")
        
        status = "healthy"
        issues = []
        
        if error_rate > self.health_thresholds['error_rate']:
            status = "unhealthy"
            issues.append(f"High error rate: {error_rate:.2%}")
        
        if avg_response_time > self.health_thresholds['response_time_ms']:
            status = "degraded" if status == "healthy" else status
            issues.append(f"High response time: {avg_response_time:.2f}ms")
        
        return {
            'status': status,
            'error_rate': error_rate,
            'avg_response_time_ms': avg_response_time,
            'issues': issues,
            'metrics': {
                'emails_processed': recent_emails,
                'processing_errors': recent_errors,
            }
        }
    
    def _check_ai_analyzer_health(self) -> Dict[str, Any]:
        """Check AI analyzer component health."""
        
        analysis_count = _metrics_collector.get_counter("ai_analysis_operations")
        analysis_errors = _metrics_collector.get_counter("ai_analysis_errors")
        
        error_rate = analysis_errors / max(1, analysis_count)
        avg_confidence = _metrics_collector.get_gauge("ai_analysis_confidence")
        
        status = "healthy"
        issues = []
        
        if error_rate > 0.05:  # 5% error rate threshold for AI
            status = "unhealthy"
            issues.append(f"High AI error rate: {error_rate:.2%}")
        
        if avg_confidence < 0.6:  # Low confidence threshold
            status = "degraded" if status == "healthy" else status
            issues.append(f"Low AI confidence: {avg_confidence:.2f}")
        
        return {
            'status': status,
            'error_rate': error_rate,
            'avg_confidence': avg_confidence,
            'issues': issues,
            'metrics': {
                'analysis_operations': analysis_count,
                'analysis_errors': analysis_errors,
            }
        }
    
    def _check_security_scanner_health(self) -> Dict[str, Any]:
        """Check security scanner component health."""
        
        scan_count = _metrics_collector.get_counter("security_scans")
        scan_errors = _metrics_collector.get_counter("security_scan_errors")
        threat_count = _metrics_collector.get_counter("security_threats_detected")
        
        error_rate = scan_errors / max(1, scan_count)
        threat_rate = threat_count / max(1, scan_count)
        
        status = "healthy"
        issues = []
        
        if error_rate > 0.02:  # 2% error rate threshold for security
            status = "unhealthy"
            issues.append(f"High security scan error rate: {error_rate:.2%}")
        
        if threat_rate > 0.5:  # More than 50% threat detection might indicate overly sensitive rules
            status = "degraded" if status == "healthy" else status
            issues.append(f"High threat detection rate: {threat_rate:.2%}")
        
        return {
            'status': status,
            'error_rate': error_rate,
            'threat_detection_rate': threat_rate,
            'issues': issues,
            'metrics': {
                'security_scans': scan_count,
                'scan_errors': scan_errors,
                'threats_detected': threat_count,
            }
        }
    
    def _check_generic_component_health(self, component_name: str) -> Dict[str, Any]:
        """Generic health check for unknown components."""
        
        return {
            'status': 'unknown',
            'message': f"No specific health check implemented for {component_name}",
            'metrics': {}
        }
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        
        component_statuses = []
        unhealthy_count = 0
        degraded_count = 0
        
        # Check all known components
        for component in ["email_processor", "ai_analyzer", "security_scanner"]:
            health = self.check_component_health(component)
            component_statuses.append({
                'component': component,
                'status': health['status'],
                'issues': health.get('issues', [])
            })
            
            if health['status'] == 'unhealthy':
                unhealthy_count += 1
            elif health['status'] == 'degraded':
                degraded_count += 1
        
        # Determine overall status
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            'overall_status': overall_status,
            'components': component_statuses,
            'summary': {
                'total_components': len(component_statuses),
                'healthy': len([c for c in component_statuses if c['status'] == 'healthy']),
                'degraded': degraded_count,
                'unhealthy': unhealthy_count,
            },
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }


class ResilienceOrchestrator:
    """Orchestrates all resilience mechanisms."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize resilience orchestrator."""
        self.config = config or {}
        
        # Initialize resilience components
        self.bulkhead = BulkheadIsolation(
            max_concurrent=self.config.get('max_concurrent', 10),
            timeout=self.config.get('operation_timeout', 30.0)
        )
        
        self.degradation = GracefulDegradation()
        self.retry = AdaptiveRetry(
            max_attempts=self.config.get('max_retry_attempts', 3),
            base_delay=self.config.get('retry_base_delay', 1.0),
            max_delay=self.config.get('retry_max_delay', 60.0)
        )
        
        self.health_check = HealthCheck()
        self.metrics = ResilienceMetrics()
        
        # Register default fallback strategies
        self._register_default_fallbacks()
        
        logger.info("ResilienceOrchestrator initialized")
    
    def _register_default_fallbacks(self):
        """Register default fallback strategies."""
        
        def simple_triage_fallback():
            """Simple fallback for email triage."""
            return {
                "category": "general",
                "priority": 5,
                "summary": "Email received - requires manual review",
                "response": "Thank you for your message. We will review it and respond shortly."
            }
        
        def basic_ai_analysis_fallback():
            """Basic fallback for AI analysis."""
            return {
                "sentiment_score": 0.0,
                "urgency_indicators": [],
                "topic_keywords": [],
                "entities": [],
                "confidence": 0.1
            }
        
        self.degradation.register_fallback("email_triage", simple_triage_fallback)
        self.degradation.register_fallback("ai_analysis", basic_ai_analysis_fallback)
    
    async def execute_resilient_operation(
        self,
        operation: Callable[[], T],
        operation_name: str,
        fallback_operation: Optional[Callable[[], T]] = None,
        retryable_exceptions: tuple = (Exception,)
    ) -> T:
        """Execute operation with full resilience mechanisms."""
        
        start_time = time.perf_counter()
        self.metrics.total_attempts += 1
        
        try:
            # Execute with bulkhead isolation and graceful degradation
            result = await self.bulkhead.execute(
                lambda: self.degradation.execute_with_fallback(
                    primary_operation=lambda: self.retry.execute_with_retry(
                        operation=operation,
                        operation_name=operation_name,
                        retryable_exceptions=retryable_exceptions
                    ),
                    operation_name=operation_name,
                    fallback_operation=fallback_operation
                )
            )
            
            self.metrics.successful_attempts += 1
            
            # Update average response time
            response_time = (time.perf_counter() - start_time) * 1000
            total_time = self.metrics.average_response_time_ms * (self.metrics.successful_attempts - 1)
            self.metrics.average_response_time_ms = (total_time + response_time) / self.metrics.successful_attempts
            
            return result
            
        except Exception as e:
            self.metrics.failed_attempts += 1
            
            logger.error(f"Resilient operation {operation_name} failed: {e}")
            _metrics_collector.increment_counter(f"resilient_operation_{operation_name}_failures")
            
            raise
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status."""
        
        return {
            'metrics': self.metrics.to_dict(),
            'bulkhead': self.bulkhead.get_status(),
            'degradation_level': self.degradation.degradation_level,
            'health': self.health_check.get_overall_health(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        }


# Global resilience orchestrator instance
resilience = ResilienceOrchestrator()
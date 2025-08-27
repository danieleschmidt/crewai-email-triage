"""Robust Intelligence Framework for Email Triage.

Enterprise-grade reliability, error handling, validation, and monitoring.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import threading
from functools import wraps

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Email validation severity levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class SecurityThreatLevel(Enum):
    """Security threat levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Email validation result."""
    is_valid: bool
    level: ValidationLevel
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_content: Optional[str] = None
    security_threats: List[Dict[str, Any]] = field(default_factory=list)
    validation_time_ms: float = 0.0
    confidence_score: float = 0.0


@dataclass
class RobustnessMetrics:
    """System robustness and reliability metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    validation_failures: int = 0
    security_blocks: int = 0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    availability: float = 0.0
    last_failure_time: Optional[float] = None
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    circuit_breaker_trips: int = 0


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    test_request_count: int = 3
    
    # State tracking
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    successful_test_count: int = 0


class RobustEmailValidator:
    """Enterprise-grade email content validator with security scanning."""
    
    def __init__(self):
        self.security_patterns = {
            'malicious_urls': [
                r'https?://[^\s]*(?:bit\.ly|tinyurl|t\.co|shortened\.link)',
                r'https?://[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',
                r'(?i)(malware|virus|trojan|phishing|scam|fraud)'
            ],
            'suspicious_attachments': [
                r'\.(?:exe|scr|bat|cmd|com|pif|vbs|js|jar|zip)(?:\s|$)',
                r'(?i)attachment.*(?:exe|scr|bat|cmd)'
            ],
            'phishing_indicators': [
                r'(?i)(?:verify|confirm|update).*(?:account|password|billing)',
                r'(?i)(?:click|visit).*(?:immediately|now|urgent)',
                r'(?i)(?:suspended|closed|expired).*account'
            ],
            'social_engineering': [
                r'(?i)(?:lottery|winner|prize|million|dollars).*(?:claim|collect)',
                r'(?i)(?:prince|inheritance|funds|transfer).*(?:millions|money)',
                r'(?i)urgent.*(?:help|assistance).*(?:money|funds)'
            ]
        }
        
        self.content_limits = {
            ValidationLevel.BASIC: {'max_length': 50000, 'max_urls': 20},
            ValidationLevel.STANDARD: {'max_length': 25000, 'max_urls': 10},
            ValidationLevel.STRICT: {'max_length': 10000, 'max_urls': 5},
            ValidationLevel.PARANOID: {'max_length': 5000, 'max_urls': 2}
        }
        
        # Compiled regex patterns for performance
        self._compiled_patterns = {}
        for category, patterns in self.security_patterns.items():
            self._compiled_patterns[category] = [re.compile(pattern) for pattern in patterns]
    
    def validate_email(self, content: str, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Comprehensive email validation with security scanning."""
        start_time = time.time()
        
        result = ValidationResult(is_valid=True, level=level)
        
        try:
            # Basic sanity checks
            if not self._basic_validation(content, result):
                result.validation_time_ms = (time.time() - start_time) * 1000
                return result
            
            # Content length and structure validation
            self._validate_content_structure(content, level, result)
            
            # Security threat detection
            self._detect_security_threats(content, result)
            
            # Content sanitization
            result.sanitized_content = self._sanitize_content(content, level)
            
            # Advanced validation for strict levels
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                self._advanced_validation(content, result)
            
            # Calculate confidence score
            result.confidence_score = self._calculate_confidence(result)
            
        except Exception as e:
            logger.error(f"Email validation failed: {e}")
            result.is_valid = False
            result.issues.append(f"Validation error: {str(e)}")
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        return result
    
    def _basic_validation(self, content: str, result: ValidationResult) -> bool:
        """Basic sanity checks."""
        if not content:
            result.is_valid = False
            result.issues.append("Empty content")
            return False
        
        if not isinstance(content, str):
            result.is_valid = False
            result.issues.append(f"Invalid content type: {type(content)}")
            return False
        
        # Check for null bytes or other dangerous characters
        if '\x00' in content:
            result.is_valid = False
            result.issues.append("Null bytes detected")
            return False
        
        return True
    
    def _validate_content_structure(self, content: str, level: ValidationLevel, result: ValidationResult):
        """Validate content structure and limits."""
        limits = self.content_limits[level]
        
        # Length validation
        if len(content) > limits['max_length']:
            result.warnings.append(f"Content length ({len(content)}) exceeds limit ({limits['max_length']})")
            if level == ValidationLevel.PARANOID:
                result.is_valid = False
                result.issues.append("Content too long for paranoid validation")
        
        # URL count validation
        url_pattern = re.compile(r'https?://[^\s]+')
        urls = url_pattern.findall(content)
        if len(urls) > limits['max_urls']:
            result.warnings.append(f"URL count ({len(urls)}) exceeds limit ({limits['max_urls']})")
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                result.is_valid = False
                result.issues.append("Too many URLs")
    
    def _detect_security_threats(self, content: str, result: ValidationResult):
        """Advanced security threat detection."""
        for threat_category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(content)
                if matches:
                    threat_level = self._assess_threat_level(threat_category, matches)
                    
                    threat_info = {
                        'category': threat_category,
                        'level': threat_level.value,
                        'matches': matches[:5],  # Limit matches shown
                        'pattern': pattern.pattern
                    }
                    result.security_threats.append(threat_info)
                    
                    if threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
                        result.is_valid = False
                        result.issues.append(f"Critical security threat: {threat_category}")
                    elif threat_level == SecurityThreatLevel.MEDIUM:
                        result.warnings.append(f"Potential security risk: {threat_category}")
    
    def _assess_threat_level(self, category: str, matches: List[str]) -> SecurityThreatLevel:
        """Assess threat level based on category and matches."""
        threat_levels = {
            'malicious_urls': SecurityThreatLevel.HIGH,
            'suspicious_attachments': SecurityThreatLevel.CRITICAL,
            'phishing_indicators': SecurityThreatLevel.HIGH,
            'social_engineering': SecurityThreatLevel.MEDIUM
        }
        
        base_level = threat_levels.get(category, SecurityThreatLevel.LOW)
        
        # Escalate based on number of matches
        if len(matches) > 5:
            if base_level == SecurityThreatLevel.MEDIUM:
                return SecurityThreatLevel.HIGH
            elif base_level == SecurityThreatLevel.HIGH:
                return SecurityThreatLevel.CRITICAL
        
        return base_level
    
    def _sanitize_content(self, content: str, level: ValidationLevel) -> str:
        """Sanitize email content based on validation level."""
        sanitized = content
        
        if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            # Remove potential script tags
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
            
            # Neutralize potentially dangerous URLs for strict levels
            if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                sanitized = re.sub(r'https?://[^\s]+', '[URL_REMOVED]', sanitized)
            
            # Remove or sanitize email addresses in paranoid mode
            if level == ValidationLevel.PARANOID:
                sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REMOVED]', sanitized)
        
        return sanitized
    
    def _advanced_validation(self, content: str, result: ValidationResult):
        """Advanced validation for strict security levels."""
        # Check for Base64 encoded content that might hide malicious code
        base64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
        base64_matches = base64_pattern.findall(content)
        if len(base64_matches) > 3:
            result.warnings.append("Multiple Base64-like strings detected")
        
        # Check for suspicious Unicode characters
        suspicious_unicode = re.compile(r'[\u200B-\u200D\u2060\uFEFF]')  # Zero-width characters
        if suspicious_unicode.search(content):
            result.warnings.append("Suspicious Unicode characters detected")
        
        # Check entropy for potential encoded content
        entropy = self._calculate_entropy(content)
        if entropy > 6.5:  # High entropy might indicate encoded content
            result.warnings.append(f"High content entropy: {entropy:.2f}")
    
    def _calculate_entropy(self, content: str) -> float:
        """Calculate Shannon entropy of content."""
        if not content:
            return 0.0
        
        # Count character frequencies
        frequencies = {}
        for char in content:
            frequencies[char] = frequencies.get(char, 0) + 1
        
        # Calculate entropy
        import math
        entropy = 0.0
        content_length = len(content)
        for count in frequencies.values():
            if count > 0:
                probability = count / content_length
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_confidence(self, result: ValidationResult) -> float:
        """Calculate validation confidence score."""
        base_confidence = 1.0 if result.is_valid else 0.0
        
        # Reduce confidence for warnings
        warning_penalty = len(result.warnings) * 0.1
        
        # Reduce confidence for security threats
        threat_penalty = 0.0
        for threat in result.security_threats:
            if threat['level'] == SecurityThreatLevel.CRITICAL.value:
                threat_penalty += 0.5
            elif threat['level'] == SecurityThreatLevel.HIGH.value:
                threat_penalty += 0.3
            elif threat['level'] == SecurityThreatLevel.MEDIUM.value:
                threat_penalty += 0.1
        
        final_confidence = base_confidence - warning_penalty - threat_penalty
        return max(0.0, min(final_confidence, 1.0))


class RobustIntelligenceFramework:
    """Enterprise robustness framework with circuit breakers, monitoring, and self-healing."""
    
    def __init__(self):
        self.metrics = RobustnessMetrics()
        self.validator = RobustEmailValidator()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="robust-intelligence")
        self._metrics_lock = threading.Lock()
        
        # Initialize default circuit breakers
        self._init_circuit_breakers()
    
    def _init_circuit_breakers(self):
        """Initialize circuit breakers for different components."""
        self.circuit_breakers = {
            'email_processing': CircuitBreaker('email_processing', failure_threshold=3),
            'validation': CircuitBreaker('validation', failure_threshold=5),
            'nlp_analysis': CircuitBreaker('nlp_analysis', failure_threshold=4),
            'security_scan': CircuitBreaker('security_scan', failure_threshold=2)
        }
    
    def robust_process_email(self, content: str, processing_func: Callable, 
                           validation_level: ValidationLevel = ValidationLevel.STANDARD,
                           timeout: float = 30.0) -> Dict[str, Any]:
        """Robustly process email with comprehensive error handling and monitoring."""
        start_time = time.time()
        request_id = self._generate_request_id(content)
        
        try:
            # Update metrics
            with self._metrics_lock:
                self.metrics.total_requests += 1
            
            # Check circuit breaker
            if not self._check_circuit_breaker('email_processing'):
                return self._create_error_response("Service temporarily unavailable", request_id)
            
            # Validate email first
            validation_result = self._validate_with_circuit_breaker(content, validation_level)
            if not validation_result.is_valid:
                with self._metrics_lock:
                    self.metrics.validation_failures += 1
                return self._create_validation_error_response(validation_result, request_id)
            
            # Process email with timeout
            try:
                future = self.thread_pool.submit(processing_func, validation_result.sanitized_content or content)
                processing_result = future.result(timeout=timeout)
                
                # Record success
                self._record_success()
                
                return {
                    'success': True,
                    'request_id': request_id,
                    'processing_result': processing_result,
                    'validation_result': self._serialize_validation_result(validation_result),
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'robustness_metrics': self._get_current_metrics()
                }
                
            except FutureTimeoutError:
                self._record_failure('email_processing', "Processing timeout")
                return self._create_error_response("Processing timeout", request_id)
            except Exception as e:
                self._record_failure('email_processing', str(e))
                return self._create_error_response(f"Processing failed: {str(e)}", request_id)
                
        except Exception as e:
            logger.error(f"Robust email processing failed: {e}")
            self._record_failure('email_processing', str(e))
            return self._create_error_response(f"Framework error: {str(e)}", request_id)
    
    def _validate_with_circuit_breaker(self, content: str, level: ValidationLevel) -> ValidationResult:
        """Validate email with circuit breaker protection."""
        if not self._check_circuit_breaker('validation'):
            # Return basic validation when circuit breaker is open
            result = ValidationResult(is_valid=True, level=ValidationLevel.BASIC)
            result.warnings.append("Validation service degraded - using basic validation")
            result.sanitized_content = content
            return result
        
        try:
            result = self.validator.validate_email(content, level)
            self._record_circuit_breaker_success('validation')
            return result
        except Exception as e:
            self._record_circuit_breaker_failure('validation', str(e))
            # Fallback to basic validation
            result = ValidationResult(is_valid=True, level=ValidationLevel.BASIC)
            result.warnings.append(f"Validation failed: {str(e)} - using basic validation")
            result.sanitized_content = content
            return result
    
    def _check_circuit_breaker(self, name: str) -> bool:
        """Check if circuit breaker allows requests."""
        if name not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[name]
        current_time = time.time()
        
        if breaker.state == CircuitBreakerState.CLOSED:
            return True
        elif breaker.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (breaker.last_failure_time and 
                current_time - breaker.last_failure_time > breaker.recovery_timeout):
                breaker.state = CircuitBreakerState.HALF_OPEN
                breaker.successful_test_count = 0
                logger.info(f"Circuit breaker {name} entering half-open state")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def _record_circuit_breaker_success(self, name: str):
        """Record successful operation for circuit breaker."""
        if name not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[name]
        
        if breaker.state == CircuitBreakerState.HALF_OPEN:
            breaker.successful_test_count += 1
            if breaker.successful_test_count >= breaker.test_request_count:
                breaker.state = CircuitBreakerState.CLOSED
                breaker.failure_count = 0
                logger.info(f"Circuit breaker {name} recovered to closed state")
        elif breaker.state == CircuitBreakerState.CLOSED:
            breaker.failure_count = max(0, breaker.failure_count - 1)
    
    def _record_circuit_breaker_failure(self, name: str, error: str):
        """Record failed operation for circuit breaker."""
        if name not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[name]
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()
        
        if breaker.failure_count >= breaker.failure_threshold:
            if breaker.state != CircuitBreakerState.OPEN:
                breaker.state = CircuitBreakerState.OPEN
                with self._metrics_lock:
                    self.metrics.circuit_breaker_trips += 1
                logger.warning(f"Circuit breaker {name} tripped: {error}")
    
    def _record_success(self):
        """Record successful request."""
        with self._metrics_lock:
            self.metrics.successful_requests += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self._update_derived_metrics()
    
    def _record_failure(self, component: str, error: str):
        """Record failed request."""
        with self._metrics_lock:
            self.metrics.failed_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()
            self._update_derived_metrics()
        
        # Update circuit breaker
        self._record_circuit_breaker_failure(component, error)
        
        logger.error(f"Component {component} failed: {error}")
    
    def _update_derived_metrics(self):
        """Update calculated metrics."""
        total = self.metrics.total_requests
        if total > 0:
            self.metrics.error_rate = self.metrics.failed_requests / total
            self.metrics.availability = self.metrics.successful_requests / total
    
    def _generate_request_id(self, content: str) -> str:
        """Generate unique request ID."""
        timestamp = str(time.time())
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        return f"req_{timestamp}_{content_hash}"
    
    def _create_error_response(self, error_message: str, request_id: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'request_id': request_id,
            'robustness_metrics': self._get_current_metrics(),
            'circuit_breaker_status': self._get_circuit_breaker_status()
        }
    
    def _create_validation_error_response(self, validation_result: ValidationResult, request_id: str) -> Dict[str, Any]:
        """Create validation error response."""
        return {
            'success': False,
            'error': 'Email validation failed',
            'request_id': request_id,
            'validation_issues': validation_result.issues,
            'validation_warnings': validation_result.warnings,
            'security_threats': validation_result.security_threats,
            'robustness_metrics': self._get_current_metrics()
        }
    
    def _serialize_validation_result(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Serialize validation result for JSON response."""
        return {
            'is_valid': validation_result.is_valid,
            'level': validation_result.level.value,
            'issues': validation_result.issues,
            'warnings': validation_result.warnings,
            'security_threats': validation_result.security_threats,
            'confidence_score': validation_result.confidence_score,
            'validation_time_ms': validation_result.validation_time_ms
        }
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current robustness metrics."""
        with self._metrics_lock:
            return {
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'error_rate': self.metrics.error_rate,
                'availability': self.metrics.availability,
                'consecutive_successes': self.metrics.consecutive_successes,
                'consecutive_failures': self.metrics.consecutive_failures,
                'circuit_breaker_trips': self.metrics.circuit_breaker_trips
            }
    
    def _get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker status for all breakers."""
        status = {}
        for name, breaker in self.circuit_breakers.items():
            status[name] = {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'failure_threshold': breaker.failure_threshold,
                'last_failure_time': breaker.last_failure_time
            }
        return status
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            'overall_health': 'healthy' if self.metrics.availability > 0.95 else 'degraded' if self.metrics.availability > 0.8 else 'unhealthy',
            'robustness_metrics': self._get_current_metrics(),
            'circuit_breakers': self._get_circuit_breaker_status(),
            'thread_pool_status': {
                'active_threads': self.thread_pool._threads,
                'queue_size': self.thread_pool._work_queue.qsize() if hasattr(self.thread_pool._work_queue, 'qsize') else 0
            }
        }


# Global robust framework instance
_robust_framework = None

def get_robust_framework() -> RobustIntelligenceFramework:
    """Get global robust intelligence framework instance."""
    global _robust_framework
    if _robust_framework is None:
        _robust_framework = RobustIntelligenceFramework()
    return _robust_framework


def robust_email_processing_decorator(validation_level: ValidationLevel = ValidationLevel.STANDARD, 
                                    timeout: float = 30.0):
    """Decorator for robust email processing with automatic error handling."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(content: str, *args, **kwargs):
            framework = get_robust_framework()
            
            def processing_func(sanitized_content: str):
                return func(sanitized_content, *args, **kwargs)
            
            return framework.robust_process_email(content, processing_func, validation_level, timeout)
        return wrapper
    return decorator